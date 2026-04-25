import OpenAI from 'openai';
import { LLMProvider } from '@inferagraph/core';
import type { LLMCompletionRequest, LLMCompletionResponse } from '@inferagraph/core';
import type { OpenAIProviderConfig } from './types.js';

const DEFAULT_MODEL = 'gpt-4o';
const DEFAULT_MAX_TOKENS = 1024;

export class OpenAIProvider extends LLMProvider {
  readonly name = 'openai';
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly maxTokens: number;

  constructor(config: OpenAIProviderConfig) {
    super();
    this.client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
      ...(config.organization ? { organization: config.organization } : {}),
    });
    this.model = config.model ?? DEFAULT_MODEL;
    this.maxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS;
  }

  async complete(request: LLMCompletionRequest): Promise<LLMCompletionResponse> {
    const messages = request.messages.map((m) => ({
      role: m.role as 'system' | 'user' | 'assistant',
      content: m.content,
    }));

    const response = await this.client.chat.completions.create({
      model: this.model,
      max_tokens: request.maxTokens ?? this.maxTokens,
      messages,
    });

    const choice = response.choices[0];
    return {
      content: choice?.message?.content ?? '',
      usage: response.usage
        ? {
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens ?? 0,
          }
        : undefined,
    };
  }

  isConfigured(): boolean {
    return true;
  }
}
