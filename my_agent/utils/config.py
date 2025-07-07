# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Configuration management for MyAgent."""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, skip loading .env file


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OPENROUTER = "openrouter"
    DOUBAO = "doubao"
    OLLAMA = "ollama"


@dataclass
class ModelParameters:
    """Parameters for LLM model configuration."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 0
    max_retries: int = 10
    parallel_tool_calls: bool = True


@dataclass
class LakeviewConfig:
    """Configuration for Lakeview summarization."""
    model_provider: str = "anthropic"
    model_name: str = "claude-3-5-sonnet-20241022"


@dataclass
class Config:
    """Main configuration class for MyAgent."""
    default_provider: str = "anthropic"
    max_steps: int = 20
    enable_lakeview: bool = True
    model_providers: Dict[str, ModelParameters] = field(default_factory=dict)
    lakeview_config: LakeviewConfig = field(default_factory=LakeviewConfig)


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file and environment variables.
    
    Args:
        config_file: Path to configuration file. Defaults to 'my_agent_config.json'
        
    Returns:
        Config object with loaded settings
    """
    if config_file is None:
        config_file = "my_agent_config.json"
    
    # Start with default config
    config_data = {
        "default_provider": "anthropic",
        "max_steps": 20,
        "enable_lakeview": True,
        "model_providers": {},
        "lakeview_config": {
            "model_provider": "anthropic",
            "model_name": "claude-3-5-sonnet-20241022"
        }
    }
    
    # Load from file if it exists
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config_data.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    # Set up default model providers with environment variables
    default_providers = {
        "openai": ModelParameters(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=128000,
            temperature=0.5,
            top_p=1.0,
            max_retries=10
        ),
        "anthropic": ModelParameters(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            top_k=0,
            max_retries=10
        ),
        "azure": ModelParameters(
            model="gpt-4o",
            api_key=os.getenv("AZURE_API_KEY"),
            base_url=os.getenv("AZURE_BASE_URL"),
            api_version="2024-03-01-preview",
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            top_k=0,
            max_retries=10
        ),
        "openrouter": ModelParameters(
            model="openai/gpt-4o",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            top_k=0,
            max_retries=10
        ),
        "doubao": ModelParameters(
            model="doubao-seed-1.6",
            api_key=os.getenv("DOUBAO_API_KEY"),
            base_url=os.getenv("DOUBAO_API_BASE_URL"),
            max_tokens=8192,
            temperature=0.5,
            top_p=1.0,
            max_retries=20
        ),
        "ollama": ModelParameters(
            model="llama3",
            base_url="http://localhost:11434",
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            max_retries=10
        )
    }
    
    # Merge with config file providers
    for provider_name, provider_config in config_data.get("model_providers", {}).items():
        if provider_name in default_providers:
            # Update existing provider with config file values
            provider_params = default_providers[provider_name]
            for key, value in provider_config.items():
                if hasattr(provider_params, key):
                    setattr(provider_params, key, value)
        else:
            # Create new provider from config
            default_providers[provider_name] = ModelParameters(**provider_config)
    
    config_data["model_providers"] = default_providers
    
    # Create lakeview config
    lakeview_data = config_data.get("lakeview_config", {})
    lakeview_config = LakeviewConfig(**lakeview_data)
    
    return Config(
        default_provider=config_data["default_provider"],
        max_steps=config_data["max_steps"],
        enable_lakeview=config_data["enable_lakeview"],
        model_providers=config_data["model_providers"],
        lakeview_config=lakeview_config
    )