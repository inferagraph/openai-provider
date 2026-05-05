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

### Azure OpenAI / OpenRouter / GitHub Models

The `baseURL` (or pre-built `client`) escape hatch covers any OpenAI-compatible API:

```ts
openaiProvider({
  apiKey: process.env.AZURE_OPENAI_KEY!,
  baseURL: 'https://my-resource.openai.azure.com/openai/deployments/my-deployment',
  model: 'gpt-4o',
});
```

### Streaming + tool calls

`stream()` translates the locked `LLMToolDefinition[]` shape into OpenAI's function-tool format and emits unified `LLMStreamEvent`s (`text` / `tool_call` / `done`). Tool calls are buffered per `index` across deltas and flushed after the text stream closes — matching the contract's documented order.

### Embeddings

`embed(texts)` returns `Vector[]` (one per input, in input order) using the configured `embeddingModel`. Empty inputs short-circuit to `[]` rather than calling the API.

## License

MIT
