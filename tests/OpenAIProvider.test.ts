import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAIProvider } from '../src/OpenAIProvider.js';

const mockCreate = vi.fn();

vi.mock('openai', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: (...args: unknown[]) => mockCreate(...args),
        },
      },
    })),
  };
});

describe('OpenAIProvider', () => {
  let provider: OpenAIProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: 'Test response about Adam.' } }],
      usage: { prompt_tokens: 50, completion_tokens: 10 },
    });
    provider = new OpenAIProvider({ apiKey: 'test-key' });
  });

  it('should have name openai', () => {
    expect(provider.name).toBe('openai');
  });

  it('should be configured', () => {
    expect(provider.isConfigured()).toBe(true);
  });

  it('should complete a request', async () => {
    const result = await provider.complete({
      messages: [
        { role: 'system', content: 'You are a Bible scholar.' },
        { role: 'user', content: 'Who is Adam?' },
      ],
    });
    expect(result.content).toBe('Test response about Adam.');
    expect(result.usage?.inputTokens).toBe(50);
    expect(result.usage?.outputTokens).toBe(10);
  });

  it('should pass maxTokens from request when provided', async () => {
    await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
      maxTokens: 512,
    });
    expect(mockCreate).toHaveBeenCalledWith(
      expect.objectContaining({ max_tokens: 512 }),
    );
  });

  it('should handle empty choices', async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [],
      usage: { prompt_tokens: 10, completion_tokens: 0 },
    });

    const result = await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('');
  });

  it('should return undefined usage when response has no usage', async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: 'hi' } }],
      usage: undefined,
    });

    const result = await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('hi');
    expect(result.usage).toBeUndefined();
  });

  it('should accept custom config with baseURL and organization', () => {
    const custom = new OpenAIProvider({
      apiKey: 'test-key',
      model: 'gpt-4-turbo',
      maxTokens: 2048,
      baseURL: 'https://custom.openai.com',
      organization: 'org-123',
    });
    expect(custom.name).toBe('openai');
  });

  describe('stream', () => {
    it('should yield text chunks and done', async () => {
      const asyncChunks = (async function* () {
        yield { choices: [{ delta: { content: 'Hello' } }] };
        yield { choices: [{ delta: { content: ' world' } }] };
      })();

      mockCreate.mockResolvedValueOnce(asyncChunks);

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'text', content: 'Hello' },
        { type: 'text', content: ' world' },
        { type: 'done', content: '' },
      ]);
    });

    it('should skip chunks with no content', async () => {
      const asyncChunks = (async function* () {
        yield { choices: [{ delta: {} }] };
        yield { choices: [{ delta: { content: 'data' } }] };
        yield { choices: [] };
      })();

      mockCreate.mockResolvedValueOnce(asyncChunks);

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'text', content: 'data' },
        { type: 'done', content: '' },
      ]);
    });

    it('should yield error chunk on failure', async () => {
      mockCreate.mockRejectedValueOnce(new Error('API rate limit'));

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'error', content: 'API rate limit' },
      ]);
    });

    it('should yield error chunk with stringified non-Error', async () => {
      mockCreate.mockRejectedValueOnce('something went wrong');

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'error', content: 'something went wrong' },
      ]);
    });

    it('should pass temperature when provided', async () => {
      const asyncChunks = (async function* () {
        yield { choices: [{ delta: { content: 'ok' } }] };
      })();

      mockCreate.mockResolvedValueOnce(asyncChunks);

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.5,
      })) {
        chunks.push(chunk);
      }

      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.5, stream: true }),
      );
    });

    it('should use request maxTokens over default', async () => {
      const asyncChunks = (async function* () {
        yield { choices: [{ delta: { content: 'ok' } }] };
      })();

      mockCreate.mockResolvedValueOnce(asyncChunks);

      for await (const _chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
        maxTokens: 256,
      })) {
        // consume
      }

      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({ max_tokens: 256 }),
      );
    });
  });
});
