# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Basic LLM types and utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..tools.base import ToolCall, ToolResult


@dataclass
class LLMUsage:
    """Token usage information from LLM response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: 'LLMUsage') -> 'LLMUsage':
        """Add two usage objects together."""
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {"role": self.role}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            
        if self.tool_result is not None:
            result["tool_result"] = self.tool_result.to_dict()
            
        if self.name is not None:
            result["name"] = self.name
            
        return result


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[LLMUsage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        result = {"content": self.content}
        
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            
        if self.usage is not None:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            }
            
        if self.model is not None:
            result["model"] = self.model
            
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
            
        return result