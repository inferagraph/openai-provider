# @inferagraph/openai-provider

OpenAI provider plugin for [@inferagraph/core](https://github.com/inferagraph/core).

## Installation

```bash
pnpm add @inferagraph/openai-provider @inferagraph/core
```

## Usage

```typescript
import { OpenAIProvider } from '@inferagraph/openai-provider';

const provider = new OpenAIProvider({
  apiKey: 'your-api-key',
  model: 'gpt-4o',
});
```

## License

MIT
