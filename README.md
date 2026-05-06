# @inferagraph/openai-provider

OpenAI provider plugin for [@inferagraph/core](https://github.com/inferagraph/core). Includes chat (`complete` + `stream` with tool calls) and embeddings (`embed`).

## Installation

```bash
pnpm add @inferagraph/openai-provider @inferagraph/core
```

## Usage

```ts
import { openaiProvider } from '@inferagraph/openai-provider';
import { InferaGraph } from '@inferagraph/core/react';

<InferaGraph
  data={data}
  llm={openaiProvider({
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'gpt-4o-mini',                   // optional; default 'gpt-4o-mini'
    embeddingModel: 'text-embedding-3-small', // optional; default 'text-embedding-3-small'
  })}
/>
```

### Configuration

| Option | Description |
|---|---|
| `apiKey` | OpenAI API key. Server-side only. Ignored when `client` is provided. |
| `model` | Chat model. Default `'gpt-4o-mini'`. |
| `embeddingModel` | Default embedding model. Default `'text-embedding-3-small'`. Per-call override via `EmbedOptions.model`. |
| `baseURL` | Override the OpenAI endpoint — works for **Azure OpenAI**, **OpenRouter**, **GitHub Models**, or any compatible API. Ignored when `client` is provided. |
| `organization` | Optional OpenAI org id. Ignored when `client` is provided. |
| `project` | Optional OpenAI project id. Ignored when `client` is provided. |
| `client` | Pre-built `OpenAI` SDK client. When supplied, all other connection fields are ignored. Primary use case: tests / mocks. |

### Azure OpenAI

Use `azureOpenaiProvider` — it encapsulates Azure OpenAI v1 SDK construction so you never build an `OpenAI` / `AzureOpenAI` client by hand:

```ts
import { azureOpenaiProvider } from '@inferagraph/openai-provider';

azureOpenaiProvider({
  endpoint: process.env.AZURE_OPENAI_ENDPOINT!, // e.g. https://my-resource.openai.azure.com/
  apiKey: process.env.AZURE_OPENAI_KEY!,
  deployment: 'gpt-4o',                          // chat deployment name
  embeddingDeployment: 'text-embedding-3-small', // optional; omit for chat-only
});
```

| Option | Description |
|---|---|
| `endpoint` | Bare resource URL. Trailing slashes are trimmed; the factory appends `/openai/v1/` for you. The v1 surface replaces dated `api-version` query strings entirely ([Azure docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)). |
| `apiKey` | Azure OpenAI API key. Server-side only. |
| `deployment` | Chat deployment name. Sent as `model` on chat completions. |
| `embeddingDeployment` | Embedding deployment name. Sent as `model` on embeddings calls. Optional — when omitted the provider has no `embed` capability. |
| `client` | Pre-built `OpenAI` SDK client. When supplied, all other connection fields are ignored. Primary use case: tests / mocks. |

The factory uses the standard `OpenAI` class (NOT `AzureOpenAI`) — the v1 endpoint is fully OpenAI-compatible.

#### Legacy escape hatch

For OpenRouter, GitHub Models, or any other OpenAI-compatible API, the `openaiProvider` `baseURL` (or pre-built `client`) escape hatch still works:

```ts
openaiProvider({
  apiKey: process.env.OTHER_PROVIDER_KEY!,
  baseURL: 'https://other-openai-compatible-host/v1',
  model: 'gpt-4o',
});
```

For Azure specifically, prefer `azureOpenaiProvider` over hand-rolling `openaiProvider({ client: new AzureOpenAI(...) })` — the new factory keeps Azure-specific URL math out of your codebase.

### Streaming + tool calls

`stream()` translates the locked `LLMToolDefinition[]` shape into OpenAI's function-tool format and emits unified `LLMStreamEvent`s (`text` / `tool_call` / `done`). Tool calls are buffered per `index` across deltas and flushed after the text stream closes — matching the contract's documented order.

### `streamMessages(messages, opts)` (recommended)

`stream(prompt: string)` accepts a single user prompt. `streamMessages(messages)` accepts a structured conversation array, which unlocks:

- **`system` role** for system prompts. Tool-use-trained models heavily discount instructions delivered as user-role content; passing them under `system` keeps directives where the model is trained to obey them. (Better than prepending to the user message.)
- **`assistant` role** to replay prior model turns — multi-turn conversation memory, corrective-retry flows after malformed tool calls, etc.
- **Multi-turn conversations** as a sequence of alternating `user` / `assistant` turns following an optional leading `system` turn.

Signature (peer dep `@inferagraph/core@^0.8.0` exports the `LLMMessage` / `LLMRole` types):

```ts
import type { LLMMessage, LLMRole } from '@inferagraph/core';

provider.streamMessages(
  messages: LLMMessage[],
  opts?: StreamOptions,
): AsyncIterable<LLMStreamEvent>;
```

Example — system prompt plus a 2-turn exchange:

```ts
import { openaiProvider } from '@inferagraph/openai-provider';
import type { LLMMessage } from '@inferagraph/core';

const provider = openaiProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'gpt-4o-mini',
});

const messages: LLMMessage[] = [
  { role: 'system', content: 'You are a concise assistant. Reply in one sentence.' },
  { role: 'user', content: 'Who wrote the Iliad?' },
  { role: 'assistant', content: 'Tradition attributes the Iliad to Homer.' },
  { role: 'user', content: 'And the Odyssey?' },
];

for await (const ev of provider.streamMessages!(messages)) {
  if (ev.type === 'text') process.stdout.write(ev.delta);
  if (ev.type === 'done') break;
}
```

The OpenAI SDK keeps `system` inline as the first message in the `chat.completions` `messages` array, so the contract array maps onto the SDK call essentially 1:1 — roles survive end-to-end.

#### Back-compat

`stream(prompt)` still works and is unchanged. It is internally a thin wrapper that calls `streamMessages([{ role: 'user', content: prompt }])`, so single-prompt behavior is identical. New consumers should prefer `streamMessages` whenever a system prompt or prior turns are involved.

### Embeddings

`embed(texts)` returns `Vector[]` (one per input, in input order) using the configured `embeddingModel`. Empty inputs short-circuit to `[]` rather than calling the API.

## License

MIT
