import OpenAI from 'openai';
import type {
  LLMProvider,
  LLMStreamEvent,
  StreamOptions,
} from '@inferagraph/core';

/**
 * A single embedding vector — a flat array of floats.
 *
 * Mirrors the locked `Vector` shape in `@inferagraph/core` so this provider
 * compiles even before core publishes the embeddings extension. Once core
 * exports the type, the structural shape is identical and consumers see
 * the same `number[]`.
 */
export type Vector = number[];

/**
 * `LLMProvider` extended with the optional `embed()` method from the locked
 * Phase 3 contract. Declared locally so this package can ship before the
 * matching core release; once core exports `embed?` on `LLMProvider`, the
 * shape is structurally identical.
 */
export interface LLMProviderWithEmbed extends LLMProvider {
  embed(texts: string[], opts?: EmbedOptions): Promise<Vector[]>;
}

/**
 * Per-call options for {@link LLMProvider.embed}.
 *
 * Mirrors the locked `EmbedOptions` contract from `@inferagraph/core`.
 *
 * `signal` is typed via `StreamOptions['signal']` so this package's tsconfig
 * does not need to pull in the DOM lib for the `AbortSignal` global.
 */
export interface EmbedOptions {
  /** Override the provider's default embedding model for this call. */
  model?: string;
  /** Optional cancellation signal; forwarded to the OpenAI SDK. */
  signal?: StreamOptions['signal'];
}

/**
 * Configuration for the OpenAI provider.
 *
 * The provider talks to the OpenAI Chat Completions API. It is server-side
 * only — the OpenAI SDK uses Node fetch / streams. Hosts that need to call
 * this from a browser bundle must polyfill or use the SDK's edge build.
 */
export interface OpenAIProviderConfig {
  /**
   * OpenAI API key. NEVER expose this to the browser; pass it in from a
   * server-side environment variable.
   *
   * Ignored when {@link client} is provided.
   */
  apiKey: string;
  /** Model name. Default `'gpt-4o-mini'`. */
  model?: string;
  /**
   * Embedding model name used by {@link LLMProvider.embed}. Default
   * `'text-embedding-3-small'`. Per-call override via {@link EmbedOptions.model}
   * takes precedence.
   */
  embeddingModel?: string;
  /**
   * Optional base URL override (Azure OpenAI, OpenRouter, GitHub Models, etc.).
   *
   * Ignored when {@link client} is provided.
   */
  baseURL?: string;
  /**
   * Pre-built OpenAI client. When supplied, {@link apiKey}, {@link baseURL},
   * {@link organization}, and {@link project} are ignored — the caller is
   * fully responsible for client configuration. Primary use case is testing
   * with a mock implementation.
   */
  client?: OpenAI;
  /** Optional organization id. Ignored when {@link client} is provided. */
  organization?: string;
  /** Optional project id. Ignored when {@link client} is provided. */
  project?: string;
}

/**
 * Construct an {@link LLMProvider} backed by the OpenAI Chat Completions API.
 *
 * Hosts pass the result to InferaGraph at construction time:
 *
 * ```ts
 * import { openaiProvider } from '@inferagraph/openai-provider';
 *
 * <InferaGraph data={data} llm={openaiProvider({ apiKey: process.env.OPENAI_KEY! })} />
 * ```
 *
 * After construction the host is hands-off; InferaGraph owns all further
 * LLM interaction.
 */
export function openaiProvider(
  config: OpenAIProviderConfig,
): LLMProviderWithEmbed {
  const client =
    config.client ??
    new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      organization: config.organization,
      project: config.project,
    });
  const model = config.model ?? 'gpt-4o-mini';
  const embeddingModel = config.embeddingModel ?? 'text-embedding-3-small';

  return {
    name: 'openai',

    async complete(prompt, opts) {
      const response = await client.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: opts?.maxTokens,
        temperature: opts?.temperature,
        response_format:
          opts?.format === 'json' ? { type: 'json_object' } : undefined,
      });
      return response.choices[0]?.message?.content ?? '';
    },

    stream(prompt, opts) {
      return openaiStream(client, model, prompt, opts);
    },

    async embed(texts: string[], opts?: EmbedOptions): Promise<Vector[]> {
      // Defensive empty-input guard: OpenAI rejects empty `input` arrays with
      // a 400. Skip the network call entirely and return [].
      if (texts.length === 0) return [];
      const response = await client.embeddings.create(
        {
          model: opts?.model ?? embeddingModel,
          input: texts,
        },
        { signal: opts?.signal },
      );
      // OpenAI returns embeddings in input order, but `index` is the
      // authoritative ordering — sort by it before mapping so we never
      // return a misaligned vector if the SDK reorders in future.
      return response.data
        .slice()
        .sort((a, b) => a.index - b.index)
        .map((d) => d.embedding);
    },
  };
}

async function* openaiStream(
  client: OpenAI,
  model: string,
  prompt: string,
  opts: StreamOptions = {},
): AsyncIterable<LLMStreamEvent> {
  // Translate the contract's neutral tool shape into OpenAI's function-tool
  // format. Empty-tools is normalized to `undefined` so the SDK doesn't
  // attach an empty `tools` array (which the API rejects).
  const tools = (opts.tools ?? []).map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));

  const stream = await client.chat.completions.create(
    {
      model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: opts.maxTokens,
      temperature: opts.temperature,
      response_format:
        opts.format === 'json' ? { type: 'json_object' } : undefined,
      tools: tools.length > 0 ? tools : undefined,
      stream: true,
    },
    { signal: opts.signal },
  );

  // Tool-call assembly: OpenAI streams tool_call deltas. We accumulate them
  // into a per-index buffer and emit one `tool_call` event per completed call
  // AFTER the text stream finishes — matches the documented contract order
  // (text, then tool calls, then done).
  const toolCallBuffers = new Map<
    number,
    { name: string; argsParts: string[] }
  >();
  let finishReason: 'stop' | 'length' | 'aborted' | undefined;

  for await (const chunk of stream) {
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta;
    if (delta?.content) {
      yield { type: 'text', delta: delta.content };
    }
    if (delta?.tool_calls) {
      for (const tc of delta.tool_calls) {
        const idx = tc.index ?? 0;
        let buf = toolCallBuffers.get(idx);
        if (!buf) {
          buf = { name: tc.function?.name ?? '', argsParts: [] };
          toolCallBuffers.set(idx, buf);
        }
        if (tc.function?.name) buf.name = tc.function.name;
        if (tc.function?.arguments) buf.argsParts.push(tc.function.arguments);
      }
    }
    if (choice.finish_reason) {
      const fr = choice.finish_reason;
      // OpenAI emits 'stop' | 'length' | 'tool_calls' | 'content_filter' |
      // 'function_call'. Only 'stop' and 'length' map directly to the
      // contract; everything else falls through to 'stop' on the final event.
      if (fr === 'stop' || fr === 'length') {
        finishReason = fr;
      }
    }
  }

  for (const buf of toolCallBuffers.values()) {
    if (!buf.name) continue;
    yield {
      type: 'tool_call',
      name: buf.name,
      arguments: buf.argsParts.join(''),
    };
  }

  yield { type: 'done', reason: finishReason ?? 'stop' };
}
