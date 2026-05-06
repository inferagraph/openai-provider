import { describe, it, expect, vi, beforeEach } from 'vitest';
import type OpenAI from 'openai';
import { azureOpenaiProvider } from '../src/index.js';

/**
 * Mock OpenAI client builder reused per test. Mirrors the shape used in
 * `openaiProvider.test.ts` so the new factory's behavior is verified against
 * the same surface area: `chat.completions.create` for chat / streaming and
 * `embeddings.create` for embeddings.
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

describe('azureOpenaiProvider', () => {
  let client: OpenAI;
  let create: ReturnType<typeof vi.fn>;
  let embedCreate: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    ({ client, create, embedCreate } = buildMockClient());
  });

  it('builds an OpenAI client with the v1 baseURL when no client is injected', () => {
    // Arrange: a sentinel ctor we control. The factory should call it with
    // { apiKey, baseURL } where baseURL is the resource endpoint plus
    // `/openai/v1/`. The v1 surface replaces dated `api-version` query strings
    // entirely (https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle).
    const seenArgs: Array<{ apiKey?: string; baseURL?: string }> = [];
    class MockOpenAI {
      chat = { completions: { create: vi.fn() } };
      embeddings = { create: vi.fn() };
      constructor(args: { apiKey?: string; baseURL?: string }) {
        seenArgs.push(args);
      }
    }

    azureOpenaiProvider({
      apiKey: 'azure-key',
      endpoint: 'https://x.openai.azure.com/',
      deployment: 'gpt-4o-chat',
      embeddingDeployment: 'text-embed-3',
      // Test seam: pass the ctor via the openaiCtor escape hatch below.
      openaiCtor: MockOpenAI as unknown as typeof OpenAI,
    });

    expect(seenArgs).toHaveLength(1);
    expect(seenArgs[0]).toEqual({
      apiKey: 'azure-key',
      baseURL: 'https://x.openai.azure.com/openai/v1/',
    });
  });

  it('trims trailing slashes from endpoint before appending /openai/v1/', () => {
    const seenArgs: Array<{ apiKey?: string; baseURL?: string }> = [];
    class MockOpenAI {
      chat = { completions: { create: vi.fn() } };
      embeddings = { create: vi.fn() };
      constructor(args: { apiKey?: string; baseURL?: string }) {
        seenArgs.push(args);
      }
    }

    azureOpenaiProvider({
      apiKey: 'k',
      endpoint: 'https://x.openai.azure.com//',
      deployment: 'd',
      openaiCtor: MockOpenAI as unknown as typeof OpenAI,
    });

    expect(seenArgs[0]?.baseURL).toBe('https://x.openai.azure.com/openai/v1/');
  });

  it('routes chat to the deployment via the model field', async () => {
    create.mockResolvedValueOnce({
      choices: [{ message: { content: 'hi' } }],
    });
    const provider = azureOpenaiProvider({
      apiKey: 'k',
      endpoint: 'https://x.openai.azure.com/',
      deployment: 'my-chat-deployment',
      client,
    });

    await provider.complete('ping');

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'my-chat-deployment' }),
    );
  });

  it('routes embeddings to embeddingDeployment when configured', async () => {
    embedCreate.mockResolvedValueOnce({
      data: [{ embedding: [0.1, 0.2], index: 0, object: 'embedding' as const }],
      model: 'my-embed-deployment',
      object: 'list' as const,
      usage: { prompt_tokens: 0, total_tokens: 0 },
    });
    const provider = azureOpenaiProvider({
      apiKey: 'k',
      endpoint: 'https://x.openai.azure.com/',
      deployment: 'd',
      embeddingDeployment: 'my-embed-deployment',
      client,
    });

    await provider.embed!(['hello']);

    expect(embedCreate).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'my-embed-deployment' }),
      expect.any(Object),
    );
  });

  it('exposes no embed capability when embeddingDeployment is omitted', () => {
    const provider = azureOpenaiProvider({
      apiKey: 'k',
      endpoint: 'https://x.openai.azure.com/',
      deployment: 'd',
      client,
    });
    expect(provider.embed).toBeUndefined();
  });
});
