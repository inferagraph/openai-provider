import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAIProvider } from '../src/OpenAIProvider.js';

vi.mock('openai', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({
            choices: [{ message: { content: 'Test response about Adam.' } }],
            usage: { prompt_tokens: 50, completion_tokens: 10 },
          }),
        },
      },
    })),
  };
});

describe('OpenAIProvider', () => {
  let provider: OpenAIProvider;

  beforeEach(() => {
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
  });

  it('should handle empty choices', async () => {
    const { default: OpenAI } = await import('openai');
    const mockInstance = new (OpenAI as any)();
    mockInstance.chat.completions.create.mockResolvedValueOnce({
      choices: [],
      usage: { prompt_tokens: 10, completion_tokens: 0 },
    });

    const customProvider = new OpenAIProvider({ apiKey: 'test-key' });
    (customProvider as any).client = mockInstance;

    const result = await customProvider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('');
  });

  it('should accept custom config', () => {
    const custom = new OpenAIProvider({
      apiKey: 'test-key',
      model: 'gpt-4-turbo',
      maxTokens: 2048,
      organization: 'org-123',
    });
    expect(custom.name).toBe('openai');
  });
});
