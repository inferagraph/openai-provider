import { describe, it, expect, vi, beforeEach } from 'vitest';
import type OpenAI from 'openai';
import type { LLMStreamEvent } from '@inferagraph/core';
import { openaiProvider } from '../src/index.js';

/**
 * Build a minimal mock OpenAI client. `create` is a vi.fn() the tests can
 * configure per-case (resolve to a non-streaming response, or to an async
 * iterable for streaming). The provider touches `client.chat.completions.create`
 * for chat and `client.embeddings.create` for embeddings, so we expose both
 * surfaces. Tests that don't exercise embeddings can ignore `embedCreate`.
 */
function buildMockClient(): {
  client: OpenAI;
  create: ReturnType<typeof vi.fn>;
  embedCreate: ReturnType<typeof vi.fn>;
} {
  const create = vi.fn();
  const embedCreate = vi.fn();
  const client = {
    chat: {
      completions: {
        create,
      },
    },
    embeddings: {
      create: embedCreate,
    },
  } as unknown as OpenAI;
  return { client, create, embedCreate };
}

/** Helper: build an async iterable from a fixed array of OpenAI-shaped chunks. */
function asyncIterableOf<T>(items: T[]): AsyncIterable<T> {
  return (async function* () {
    for (const it of items) yield it;
  })();
}

/** Helper: drain an LLMStreamEvent async iterable into an array. */
async function collect(
  stream: AsyncIterable<LLMStreamEvent>,
): Promise<LLMStreamEvent[]> {
  const out: LLMStreamEvent[] = [];
  for await (const ev of stream) out.push(ev);
  return out;
}

describe('openaiProvider', () => {
  let client: OpenAI;
  let create: ReturnType<typeof vi.fn>;
  let embedCreate: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    ({ client, create, embedCreate } = buildMockClient());
  });

  describe('factory', () => {
    it('exposes name "openai"', () => {
      const provider = openaiProvider({ apiKey: 'k', client });
      expect(provider.name).toBe('openai');
    });

    it('uses the provided client when given (apiKey ignored)', async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: 'hi' } }],
      });
      const provider = openaiProvider({ apiKey: 'unused', client });
      const result = await provider.complete('ping');
      expect(result).toBe('hi');
      expect(create).toHaveBeenCalledOnce();
    });
  });

  describe('complete()', () => {
    it('sends the prompt as a user message and returns content', async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: 'hello world' } }],
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.complete('Who is Adam?');
      expect(result).toBe('hello world');
      expect(create).toHaveBeenCalledWith({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Who is Adam?' }],
        max_tokens: undefined,
        temperature: undefined,
        response_format: undefined,
      });
    });

    it('uses a custom model when configured', async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: 'ok' } }],
      });
      const provider = openaiProvider({
        apiKey: 'k',
        model: 'gpt-4o',
        client,
      });
      await provider.complete('x');
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'gpt-4o' }),
      );
    });

    it('passes maxTokens and temperature when provided', async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: '' } }],
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      await provider.complete('x', { maxTokens: 256, temperature: 0.3 });
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ max_tokens: 256, temperature: 0.3 }),
      );
    });

    it("forwards format='json' as response_format json_object", async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: '{}' } }],
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      await provider.complete('x', { format: 'json' });
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ response_format: { type: 'json_object' } }),
      );
    });

    it("does NOT set response_format when format='text'", async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: 'plain' } }],
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      await provider.complete('x', { format: 'text' });
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ response_format: undefined }),
      );
    });

    it('returns empty string when choices is empty', async () => {
      create.mockResolvedValueOnce({ choices: [] });
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.complete('x');
      expect(result).toBe('');
    });

    it('returns empty string when message content is null', async () => {
      create.mockResolvedValueOnce({
        choices: [{ message: { content: null } }],
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.complete('x');
      expect(result).toBe('');
    });
  });

  describe('stream()', () => {
    it('yields text events for content deltas, then a done event', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          { choices: [{ delta: { content: 'Hello' } }] },
          { choices: [{ delta: { content: ' world' } }] },
          { choices: [{ delta: {}, finish_reason: 'stop' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events).toEqual([
        { type: 'text', delta: 'Hello' },
        { type: 'text', delta: ' world' },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('emits done with reason "length" when finish_reason is length', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          { choices: [{ delta: { content: 'partial' } }] },
          { choices: [{ delta: {}, finish_reason: 'length' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.at(-1)).toEqual({ type: 'done', reason: 'length' });
    });

    it('falls back to reason "stop" when no finish_reason was sent', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([{ choices: [{ delta: { content: 'a' } }] }]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.at(-1)).toEqual({ type: 'done', reason: 'stop' });
    });

    it('falls back to "stop" when finish_reason is an exotic value (e.g. tool_calls)', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      function: { name: 'apply_filter', arguments: '{}' },
                    },
                  ],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.at(-1)).toEqual({ type: 'done', reason: 'stop' });
    });

    it('skips chunks with no choices and chunks with empty delta', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          { choices: [] },
          { choices: [{ delta: {} }] },
          { choices: [{ delta: { content: 'x' } }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events).toEqual([
        { type: 'text', delta: 'x' },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('accumulates a single tool_call across multiple delta chunks and emits AFTER text', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          { choices: [{ delta: { content: 'thinking...' } }] },
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      function: { name: 'apply_filter', arguments: '{"era"' },
                    },
                  ],
                },
              },
            ],
          },
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, function: { arguments: ':"patriarchs"}' } },
                  ],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('show patriarchs'));
      expect(events).toEqual([
        { type: 'text', delta: 'thinking...' },
        {
          type: 'tool_call',
          name: 'apply_filter',
          arguments: '{"era":"patriarchs"}',
        },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('handles multiple parallel tool_calls keyed by index', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      function: { name: 'highlight', arguments: '{"id":"a"}' },
                    },
                    {
                      index: 1,
                      function: { name: 'focus', arguments: '{"id":"b"}' },
                    },
                  ],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('do both'));
      const toolCalls = events.filter((e) => e.type === 'tool_call');
      expect(toolCalls).toEqual([
        { type: 'tool_call', name: 'highlight', arguments: '{"id":"a"}' },
        { type: 'tool_call', name: 'focus', arguments: '{"id":"b"}' },
      ]);
    });

    it('drops tool_call buffers that never received a name', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    { index: 0, function: { arguments: '{"orphan":true}' } },
                  ],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: 'stop' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('x'));
      expect(events.find((e) => e.type === 'tool_call')).toBeUndefined();
    });

    it('translates StreamOptions.tools into OpenAI function-tool format', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await collect(
        provider.stream('hi', {
          tools: [
            {
              name: 'apply_filter',
              description: 'Restrict the visible set',
              parameters: {
                type: 'object',
                properties: { predicate: { type: 'string' } },
                required: ['predicate'],
              },
            },
          ],
        }),
      );
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({
          tools: [
            {
              type: 'function',
              function: {
                name: 'apply_filter',
                description: 'Restrict the visible set',
                parameters: {
                  type: 'object',
                  properties: { predicate: { type: 'string' } },
                  required: ['predicate'],
                },
              },
            },
          ],
        }),
        expect.any(Object),
      );
    });

    it('omits tools entirely when none are supplied', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await collect(provider.stream('hi'));
      const args = create.mock.calls[0]![0] as { tools?: unknown };
      expect(args.tools).toBeUndefined();
    });

    it('forwards the AbortSignal as a request option', async () => {
      const ctrl = new AbortController();
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await collect(provider.stream('hi', { signal: ctrl.signal }));
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ stream: true }),
        { signal: ctrl.signal },
      );
    });

    it('passes maxTokens, temperature, and format=json into the streaming request', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await collect(
        provider.stream('hi', {
          maxTokens: 128,
          temperature: 0.7,
          format: 'json',
        }),
      );
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 128,
          temperature: 0.7,
          response_format: { type: 'json_object' },
          stream: true,
        }),
        expect.any(Object),
      );
    });

    it("does NOT set response_format when format='text' on stream", async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await collect(provider.stream('hi', { format: 'text' }));
      const args = create.mock.calls[0]![0] as { response_format?: unknown };
      expect(args.response_format).toBeUndefined();
    });

    it('handles a stream that yields nothing at all and still emits done', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events).toEqual([{ type: 'done', reason: 'stop' }]);
    });

    it('treats a missing tool_call index as 0 (single-tool fallback)', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      function: { name: 'highlight', arguments: '{"id":"x"}' },
                    },
                  ],
                },
              },
            ],
          },
          { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('x'));
      const toolCalls = events.filter((e) => e.type === 'tool_call');
      expect(toolCalls).toEqual([
        { type: 'tool_call', name: 'highlight', arguments: '{"id":"x"}' },
      ]);
    });
  });

  describe('embed()', () => {
    /** Minimal helper to build the OpenAI embeddings response shape. */
    function embedResponse(
      vectors: number[][],
    ): {
      data: Array<{ embedding: number[]; index: number; object: 'embedding' }>;
      model: string;
      object: 'list';
      usage: { prompt_tokens: number; total_tokens: number };
    } {
      return {
        data: vectors.map((embedding, index) => ({
          embedding,
          index,
          object: 'embedding' as const,
        })),
        model: 'text-embedding-3-small',
        object: 'list' as const,
        usage: { prompt_tokens: 0, total_tokens: 0 },
      };
    }

    it('uses the default model "text-embedding-3-small" when no opts', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[0.1, 0.2]]));
      const provider = openaiProvider({ apiKey: 'k', client });
      // embed is optional on the contract; it is always present on this provider.
      const result = await provider.embed!(['hello']);
      expect(result).toEqual([[0.1, 0.2]]);
      expect(embedCreate).toHaveBeenCalledWith(
        { model: 'text-embedding-3-small', input: ['hello'] },
        { signal: undefined },
      );
    });

    it('honours config.embeddingModel as the constructor-level default', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[1, 2, 3]]));
      const provider = openaiProvider({
        apiKey: 'k',
        client,
        embeddingModel: 'text-embedding-3-large',
      });
      await provider.embed!(['x']);
      expect(embedCreate).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'text-embedding-3-large' }),
        expect.any(Object),
      );
    });

    it('honours opts.model as a per-call override', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[0]]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await provider.embed!(['x'], { model: 'text-embedding-ada-002' });
      expect(embedCreate).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'text-embedding-ada-002' }),
        expect.any(Object),
      );
    });

    it('per-call opts.model beats config.embeddingModel', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[0]]));
      const provider = openaiProvider({
        apiKey: 'k',
        client,
        embeddingModel: 'text-embedding-3-large',
      });
      await provider.embed!(['x'], { model: 'text-embedding-3-small' });
      expect(embedCreate).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'text-embedding-3-small' }),
        expect.any(Object),
      );
    });

    it('returns a single vector for a single text input', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[0.5, -0.5, 1.0]]));
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.embed!(['only one']);
      expect(result).toEqual([[0.5, -0.5, 1.0]]);
    });

    it('returns a batch of vectors preserving input order', async () => {
      embedCreate.mockResolvedValueOnce(
        embedResponse([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]),
      );
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.embed!(['a', 'b', 'c']);
      expect(result).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      expect(embedCreate).toHaveBeenCalledWith(
        expect.objectContaining({ input: ['a', 'b', 'c'] }),
        expect.any(Object),
      );
    });

    it('reorders by `index` so a shuffled SDK response still aligns with input order', async () => {
      // Construct a deliberately scrambled response: index 2 first, then 0, then 1.
      embedCreate.mockResolvedValueOnce({
        data: [
          { embedding: [0, 0, 1], index: 2, object: 'embedding' as const },
          { embedding: [1, 0, 0], index: 0, object: 'embedding' as const },
          { embedding: [0, 1, 0], index: 1, object: 'embedding' as const },
        ],
        model: 'text-embedding-3-small',
        object: 'list' as const,
        usage: { prompt_tokens: 0, total_tokens: 0 },
      });
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.embed!(['a', 'b', 'c']);
      expect(result).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    it('forwards opts.signal as the SDK request option', async () => {
      const ctrl = new AbortController();
      embedCreate.mockResolvedValueOnce(embedResponse([[0]]));
      const provider = openaiProvider({ apiKey: 'k', client });
      await provider.embed!(['x'], { signal: ctrl.signal });
      expect(embedCreate).toHaveBeenCalledWith(expect.any(Object), {
        signal: ctrl.signal,
      });
    });

    it('returns [] for empty input WITHOUT calling the SDK', async () => {
      const provider = openaiProvider({ apiKey: 'k', client });
      const result = await provider.embed!([]);
      expect(result).toEqual([]);
      expect(embedCreate).not.toHaveBeenCalled();
    });

    it('uses the user-supplied client for embeddings (not a freshly-built one)', async () => {
      embedCreate.mockResolvedValueOnce(embedResponse([[0.42]]));
      const provider = openaiProvider({ apiKey: 'unused', client });
      const result = await provider.embed!(['ping']);
      expect(result).toEqual([[0.42]]);
      expect(embedCreate).toHaveBeenCalledOnce();
    });

    it('propagates SDK errors to the caller', async () => {
      embedCreate.mockRejectedValueOnce(new Error('rate limit'));
      const provider = openaiProvider({ apiKey: 'k', client });
      await expect(provider.embed!(['x'])).rejects.toThrow('rate limit');
    });
  });
});
