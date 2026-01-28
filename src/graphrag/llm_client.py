"""
LLM Client abstraction for report generation.

Supports:
1. DeepSeek API (OpenAI-compatible)
2. Local Qwen (via transformers or vLLM)

Design: Abstract base class allows easy switching between providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from loguru import logger


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        Multi-turn chat interface.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant response
        """
        pass


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client (OpenAI-compatible).

    DeepSeek provides high-quality Chinese+English models
    at competitive pricing via OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model name (deepseek-chat, deepseek-coder)
            base_url: API base URL
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required")

        self.model = model
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized DeepSeek client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Generate text from single prompt."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature, max_tokens, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Multi-turn chat."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


class LocalQwenClient(LLMClient):
    """
    Local Qwen model client.

    Runs Qwen models locally for privacy-sensitive applications.
    Supports multiple backends: transformers, vLLM, llama.cpp
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "transformers",
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Initialize local Qwen client.

        Args:
            model_path: HuggingFace model ID or local path
            backend: 'transformers', 'vllm', or 'llamacpp'
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            load_in_4bit: Use 4-bit quantization to save memory
        """
        self.model_path = model_path
        self.backend = backend
        self.model = None
        self.tokenizer = None

        if backend == "transformers":
            self._init_transformers(model_path, device, load_in_4bit)
        elif backend == "vllm":
            self._init_vllm(model_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(f"Initialized local Qwen: {model_path} ({backend})")

    def _init_transformers(
        self, model_path: str, device: str, load_in_4bit: bool
    ) -> None:
        """Initialize with HuggingFace transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers package required")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Quantization config for memory efficiency
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map=device,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, loading full precision")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map=device
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device
            )

    def _init_vllm(self, model_path: str) -> None:
        """Initialize with vLLM for faster inference."""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vllm package required: pip install vllm")

        self.model = LLM(model=model_path)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Generate text from single prompt."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature, max_tokens, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Multi-turn chat."""
        if self.backend == "transformers":
            return self._chat_transformers(messages, temperature, max_tokens)
        elif self.backend == "vllm":
            return self._chat_vllm(messages, temperature, max_tokens)

    def _chat_transformers(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Chat using transformers."""
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return response

    def _chat_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Chat using vLLM."""
        from vllm import SamplingParams

        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.model.generate([prompt], params)

        return outputs[0].outputs[0].text


def get_llm_client(config: Dict) -> LLMClient:
    """
    Factory function to create LLM client from config.

    Args:
        config: Dictionary with:
            - type: 'deepseek' or 'local_qwen'
            - api_key: For DeepSeek
            - model: Model name
            - model_path: For local models

    Returns:
        Configured LLM client
    """
    client_type = config.get("type", "deepseek")

    if client_type == "deepseek":
        return DeepSeekClient(
            api_key=config.get("api_key"),
            model=config.get("model", "deepseek-chat"),
        )
    elif client_type == "local_qwen":
        return LocalQwenClient(
            model_path=config.get("model_path", "Qwen/Qwen2.5-7B-Instruct"),
            backend=config.get("backend", "transformers"),
            device=config.get("device", "auto"),
            load_in_4bit=config.get("load_in_4bit", True),
        )
    else:
        raise ValueError(f"Unknown LLM client type: {client_type}")
