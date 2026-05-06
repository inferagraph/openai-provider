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

### Embeddings

`embed(texts)` returns `Vector[]` (one per input, in input order) using the configured `embeddingModel`. Empty inputs short-circuit to `[]` rather than calling the API.

## License

MIT
