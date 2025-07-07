# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""LLM client for communicating with different providers."""

import json
from typing import List, Optional

from .config import LLMProvider, ModelParameters
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from ..tools.base import Tool, ToolCall


class LLMClient:
    """Client for communicating with LLM providers."""
    
    def __init__(self, provider: str, model_params: ModelParameters):
        self.provider = LLMProvider(provider)
        self.model_params = model_params
        self.trajectory_recorder = None
        
        # Initialize the appropriate client
        if self.provider == LLMProvider.OPENAI:
            import openai
            self.client = openai.OpenAI(api_key=model_params.api_key)
        elif self.provider == LLMProvider.ANTHROPIC:
            import anthropic
            self.client = anthropic.Anthropic(api_key=model_params.api_key)
        elif self.provider == LLMProvider.AZURE:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=model_params.api_key,
                azure_endpoint=model_params.base_url,
                api_version=model_params.api_version
            )
        elif self.provider == LLMProvider.OPENROUTER:
            import openai
            self.client = openai.OpenAI(
                api_key=model_params.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif self.provider == LLMProvider.DOUBAO:
            import openai
            self.client = openai.OpenAI(
                api_key=model_params.api_key,
                base_url=model_params.base_url
            )
        elif self.provider == LLMProvider.OLLAMA:
            import openai
            self.client = openai.OpenAI(
                api_key="ollama",  # Ollama doesn't require a real API key
                base_url=model_params.base_url or "http://localhost:11434/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def set_trajectory_recorder(self, recorder):
        """Set the trajectory recorder."""
        self.trajectory_recorder = recorder
    
    def chat(self, messages: List[LLMMessage], model_params: ModelParameters, tools: List[Tool]) -> LLMResponse:
        """Send a chat request to the LLM provider."""
        
        # Record the request if trajectory recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_request(
                provider=self.provider.value,
                model=model_params.model,
                messages=[msg.to_dict() for msg in messages],
                tools=[tool.json_definition() for tool in tools] if tools else None
            )
        
        if self.provider == LLMProvider.ANTHROPIC:
            return self._chat_anthropic(messages, model_params, tools)
        else:
            return self._chat_openai_compatible(messages, model_params, tools)
    
    def _chat_anthropic(self, messages: List[LLMMessage], model_params: ModelParameters, tools: List[Tool]) -> LLMResponse:
        """Handle Anthropic API calls."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "user":
                if msg.tool_result:
                    # Handle tool result
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.call_id,
                                "content": msg.tool_result.result or msg.tool_result.error or ""
                            }
                        ]
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
            elif msg.role == "assistant":
                if msg.tool_calls:
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tool_call in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
        
        # Prepare tools for Anthropic
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for tool in tools:
                schema = tool.json_definition()
                anthropic_tools.append({
                    "name": schema["name"],
                    "description": schema["description"],
                    "input_schema": schema["parameters"]
                })
        
        # Make the API call
        kwargs = {
            "model": model_params.model,
            "max_tokens": model_params.max_tokens,
            "temperature": model_params.temperature,
            "top_p": model_params.top_p,
            "messages": anthropic_messages
        }
        
        if system_message:
            kwargs["system"] = system_message
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if model_params.top_k > 0:
            kwargs["top_k"] = model_params.top_k
        
        response = self.client.messages.create(**kwargs)
        
        # Parse response
        content = ""
        tool_calls = []
        
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=content_block.name,
                    call_id=content_block.id,
                    arguments=content_block.input,
                    id=content_block.id
                ))
        
        usage = LLMUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )
        
        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason
        )
        
        # Record the response
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_response(llm_response.to_dict())
        
        return llm_response
    
    def _chat_openai_compatible(self, messages: List[LLMMessage], model_params: ModelParameters, tools: List[Tool]) -> LLMResponse:
        """Handle OpenAI-compatible API calls."""
        # Convert messages to OpenAI format
        openai_messages = []
        
        for msg in messages:
            if msg.tool_result:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_result.call_id,
                    "content": msg.tool_result.result or msg.tool_result.error or ""
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": [tc.to_dict() for tc in msg.tool_calls]
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Prepare tools for OpenAI
        openai_tools = None
        if tools:
            openai_tools = [tool.json_definition() for tool in tools]
        
        # Make the API call
        kwargs = {
            "model": model_params.model,
            "messages": openai_messages,
            "max_tokens": model_params.max_tokens,
            "temperature": model_params.temperature,
            "top_p": model_params.top_p
        }
        
        if openai_tools:
            kwargs["tools"] = openai_tools
            if model_params.parallel_tool_calls:
                kwargs["parallel_tool_calls"] = True
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Parse response
        message = response.choices[0].message
        content = message.content or ""
        
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    call_id=tc.id,
                    arguments=json.loads(tc.function.arguments),
                    id=tc.id
                ))
        
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason
        )
        
        # Record the response
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_response(llm_response.to_dict())
        
        return llm_response