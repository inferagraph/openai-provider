import { describe, it, expect, vi, beforeEach } from 'vitest';
import type OpenAI from 'openai';
import type { LLMMessage, LLMStreamEvent } from '@inferagraph/core/data';
import { openaiProvider } from '../src/index.js';

/**
 * PROVIDER PARITY CONTRACT — OpenAI side.
 *
 * The sibling repo `@inferagraph/anthropic-provider` carries an identically-
 * shaped `tests/parity.test.ts` asserting the SAME canonical fixture shapes
 * defined below. The intent is that engine-level fallback logic in
 * `@inferagraph/core@^0.8.0`'s `emitWithFallbacks` reducer can react
 * uniformly to provider failure modes — whichever LLM is plugged in, the
 * neutral `LLMStreamEvent` sequence the engine consumes has the same shape.
 *
 * Each scenario below uses an SDK stub that emits a known malformation. We
 * collect the resulting `LLMStreamEvent[]` and assert against the canonical
 * fixture. If a real divergence between providers exists for one of these
 * shapes, this test fails on one side and the surface area of the bug is
 * surgical — fix the smaller-diff provider rather than mutating the contract.
 *
 * CANONICAL FIXTURES (must match the anthropic repo verbatim):
 *
 *   1. EMPTY_COMPLETION
 *      [{ type: 'done', reason: 'stop' }]
 *      Provider closes the stream with zero text/tool events. Engine sees
 *      a single terminal `done` and can synthesize a fallback message.
 *
 *   2. EMPTY_TOOL_CALL_OBJECT
 *      [{ type: 'tool_call', name: 'highlight', arguments: '{"ids":{}}' },
 *       { type: 'done', reason: 'stop' }]
 *      Model emits a tool_use whose JSON arguments contain `"ids":{}`
 *      (object instead of array). Provider passes the raw JSON through
 *      verbatim — the engine, not the provider, decides how to react.
 *
 *   3. EMPTY_TOOL_CALL_ARRAY
 *      [{ type: 'tool_call', name: 'highlight', arguments: '{"ids":[]}' },
 *       { type: 'done', reason: 'stop' }]
 *      Model emits a tool_use with `"ids":[]` (empty array). Same passthrough
 *      contract as the object-shape variant.
 *
 *   4. NULL_ARGUMENTS
 *      [{ type: 'tool_call', name: 'highlight', arguments: '' },
 *       { type: 'done', reason: 'stop' }]
 *      Model emits a tool_use with no arguments delta at all. Provider
 *      surfaces an empty-string `arguments` (the join of zero parts), NOT
 *      a synthetic object. Engine handles parse failure downstream.
 *
 *   5. MID_STREAM_ERROR
 *      Stream throws partway through. Provider propagates the error to the
 *      consumer (for-await-of rethrows). The collected prefix MAY contain
 *      text events emitted before the throw; no `done` event is emitted on
 *      the error path because the iterator never reached the post-loop
 *      yield. The engine wraps the throw in its own `done.error`.
 *
 *   6. MULTI_ROLE_PRESERVED
 *      streamMessages([{role:'system',...},{role:'user',...}]) preserves
 *      both roles in the SDK call body even when the stream emits zero
 *      events. The collected event sequence is [{type:'done',reason:'stop'}].
 */

/** Build a minimal mock OpenAI client (matches openaiProvider.test.ts). */
function buildMockClient(): {
  client: OpenAI;
  create: ReturnType<typeof vi.fn>;
} {
  const create = vi.fn();
  const client = {
    chat: {
      completions: { create },
    },
    embeddings: { create: vi.fn() },
  } as unknown as OpenAI;
  return { client, create };
}

function asyncIterableOf<T>(items: T[]): AsyncIterable<T> {
  return (async function* () {
    for (const it of items) yield it;
  })();
}

async function collect(
  stream: AsyncIterable<LLMStreamEvent>,
): Promise<LLMStreamEvent[]> {
  const out: LLMStreamEvent[] = [];
  for await (const ev of stream) out.push(ev);
  return out;
}

describe('openaiProvider — parity contract', () => {
  let client: OpenAI;
  let create: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    ({ client, create } = buildMockClient());
  });

  it('1. emits zero text events when SDK closes connection without payload (EMPTY_COMPLETION)', async () => {
    create.mockResolvedValueOnce(asyncIterableOf([]));
    const provider = openaiProvider({ apiKey: 'k', client });
    const events = await collect(provider.stream('hi'));
    expect(events).toEqual([{ type: 'done', reason: 'stop' }]);
  });

  it('2. forwards a malformed tool_use with {"ids":{}} as-is to the engine (EMPTY_TOOL_CALL_OBJECT)', async () => {
    // Model emitted `highlight({"ids": {}})` — arguments is a JSON object
    // whose `ids` field is an object instead of the expected array. The
    // provider MUST pass the raw JSON string through; engine logic decides
    // whether to retry, substitute, or surface as a tool error.
    create.mockResolvedValueOnce(
      asyncIterableOf([
        {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: {
                      name: 'highlight',
                      arguments: '{"ids":{}}',
                    },
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
    const events = await collect(provider.stream('who is in eden'));
    expect(events).toEqual([
      { type: 'tool_call', name: 'highlight', arguments: '{"ids":{}}' },
      { type: 'done', reason: 'stop' },
    ]);
  });

  it('3. forwards a malformed tool_use with {"ids":[]} as-is (EMPTY_TOOL_CALL_ARRAY)', async () => {
    create.mockResolvedValueOnce(
      asyncIterableOf([
        {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: {
                      name: 'highlight',
                      arguments: '{"ids":[]}',
                    },
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
    const events = await collect(provider.stream('who is in eden'));
    expect(events).toEqual([
      { type: 'tool_call', name: 'highlight', arguments: '{"ids":[]}' },
      { type: 'done', reason: 'stop' },
    ]);
  });

  it('4. forwards a tool_use with no arguments delta as empty-string arguments (NULL_ARGUMENTS)', async () => {
    // The model declared the tool but never streamed any argument JSON.
    // OpenAI delivers this as a single delta with `function.name` set and
    // `function.arguments` absent. The provider's accumulator joins zero
    // parts, yielding `arguments: ''` — engine handles the empty payload.
    create.mockResolvedValueOnce(
      asyncIterableOf([
        {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: { name: 'highlight' },
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
    const events = await collect(provider.stream('who is in eden'));
    expect(events).toEqual([
      { type: 'tool_call', name: 'highlight', arguments: '' },
      { type: 'done', reason: 'stop' },
    ]);
  });

  it('5. propagates a mid-stream provider error to the consumer (MID_STREAM_ERROR)', async () => {
    // After yielding one text delta the underlying SDK iterator throws.
    // for-await-of in the provider rethrows; the consumer sees the error
    // BEFORE the post-loop `done` yield runs. This matches the Anthropic
    // provider's behavior — neither provider swallows mid-stream errors.
    async function* throwingStream(): AsyncIterable<unknown> {
      yield { choices: [{ delta: { content: 'partial' } }] };
      throw new Error('upstream timeout');
    }
    create.mockResolvedValueOnce(throwingStream());
    const provider = openaiProvider({ apiKey: 'k', client });
    const collected: LLMStreamEvent[] = [];
    let caught: unknown;
    try {
      for await (const ev of provider.stream('hi')) {
        collected.push(ev);
      }
    } catch (err) {
      caught = err;
    }
    expect(collected).toEqual([{ type: 'text', delta: 'partial' }]);
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe('upstream timeout');
  });

  it('6. streamMessages with [system,user] preserves role separation when SDK emits trivial output (MULTI_ROLE_PRESERVED)', async () => {
    create.mockResolvedValueOnce(asyncIterableOf([]));
    const provider = openaiProvider({ apiKey: 'k', client });
    const messages: LLMMessage[] = [
      { role: 'system', content: 'You answer about scripture.' },
      { role: 'user', content: 'who lived in eden' },
    ];
    const events = await collect(provider.streamMessages!(messages));
    expect(events).toEqual([{ type: 'done', reason: 'stop' }]);
    const args = create.mock.calls[0]![0] as {
      messages: Array<{ role: string; content: string }>;
    };
    // Both roles must reach the SDK with their identities intact — system
    // does NOT collapse into user. Anthropic moves system into a top-level
    // `system` field; OpenAI keeps it inline. The contract is "no role is
    // dropped or rewritten," which both providers honor in their own way.
    expect(args.messages).toEqual([
      { role: 'system', content: 'You answer about scripture.' },
      { role: 'user', content: 'who lived in eden' },
    ]);
  });
});
