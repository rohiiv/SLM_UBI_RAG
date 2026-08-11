"""
Banking RAG Fine-tuned Qwen SLM Generator module.

Implements BaseLLMGenerator interface for text generation using Qwen fine-tuned Banking SLM.
"""

import re
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from banking_rag.config import ModelConfig, get_config
from banking_rag.constants import CANARY_TOKEN_PREFIX
from banking_rag.exceptions import LLMGenerationError
from banking_rag.utils.logger import get_logger

logger = get_logger("llm.generator")


class BaseLLMGenerator(ABC):
    """Abstract Interface for LLM / SLM Generators."""

    @abstractmethod
    def generate(self, prompt_payload: Dict[str, str]) -> str:
        """Generates text response from system and user prompt dictionary.

        Args:
            prompt_payload: Dictionary with 'system' and 'user' keys.

        Returns:
            Generated response string.

        Raises:
            LLMGenerationError: If inference fails.
        """
        pass


class QwenBankingSLMGenerator(BaseLLMGenerator):
    """Generator implementation targeting fine-tuned Qwen3-4B Banking SLM model.

    Both the model weights and the tokenizer are process-wide singletons: a class-level cache
    keyed by (model_name, device) ensures they are loaded from disk only once per process,
    and every QwenBankingSLMGenerator instance reuses the same in-memory objects.
    """

    # Shared across ALL instances of this class. Maps (model_name, device) -> (tokenizer, model).
    _model_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initializes generator with model config settings.

        Args:
            config: Optional ModelConfig object.
        """
        self.config = config or get_config().model
        self.model_name = self.config.slm_model_name
        self.model_revision = self.config.slm_model_revision
        self.device = self.config.device
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        self.max_new_tokens = self.config.max_new_tokens
        
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for HuggingFace AutoModelForCausalLM and AutoTokenizer, backed by a shared cache."""
        if self._model is not None:
            return

        cache_key = (self.model_name, self.device)

        with QwenBankingSLMGenerator._cache_lock:
            cached = QwenBankingSLMGenerator._model_cache.get(cache_key)
            if cached is not None:
                self._tokenizer, self._model = cached
                logger.info(f"Reusing already-loaded Qwen SLM '{self.model_name}' and tokenizer from shared cache.")
                return

            if not self.model_revision:
                logger.warning(
                    f"Loading SLM '{self.model_name}' with NO pinned revision. Set "
                    f"SLM_MODEL_REVISION to a specific commit SHA before real data is loaded - "
                    f"this is the generator model, so an unpinned supply-chain compromise here "
                    f"has the most direct path to influencing what the assistant tells a user."
                )
            logger.info(f"Loading Fine-tuned Banking SLM '{self.model_name}' (revision={self.model_revision or 'UNPINNED'}) on device '{self.device}'...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True, revision=self.model_revision or None,
                )
                device_map = "auto" if self.device == "cuda" else None
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=device_map,
                    trust_remote_code=True,
                    revision=self.model_revision or None,
                )
                if self.device != "cpu" and device_map is None:
                    model = model.to(self.device)
                logger.info(f"Successfully loaded fine-tuned Qwen Banking SLM model weights on {self.device}.")

            except ImportError:
                logger.warning("transformers/torch not installed. Using mock LLM response generator.")
                tokenizer = "MOCK"
                model = "MOCK"
            except Exception as e:
                logger.error(f"Failed to load Qwen SLM model {self.model_name}: {str(e)}")
                raise LLMGenerationError(f"Model initialization error: {str(e)}")

            QwenBankingSLMGenerator._model_cache[cache_key] = (tokenizer, model)
            self._tokenizer = tokenizer
            self._model = model

    def preload(self) -> None:
        """Forces the Qwen model and tokenizer to load immediately instead of lazily on first use.

        Intended to be called once at application startup so model loading happens
        deterministically before the first query is served.
        """
        self._load_model()

    def generate(self, prompt_payload: Dict[str, str]) -> str:
        """Executes text generation using Qwen SLM model.

        Args:
            prompt_payload: Dictionary containing 'system' and 'user' prompts.

        Returns:
            Generated response string with citations.

        Raises:
            LLMGenerationError: If inference fails.
        """
        system_prompt = prompt_payload.get("system", "")
        user_prompt = prompt_payload.get("user", "")
        canary_token = prompt_payload.get("canary_token")

        if not user_prompt:
            raise LLMGenerationError("User prompt cannot be empty.")

        logger.info(f"Executing Qwen SLM generation (temp={self.temperature}, max_tokens={self.max_new_tokens})")

        self._load_model()

        try:
            if self._model == "MOCK":
                logger.debug("Generating mock compliance officer response.")
                mock_text = (
                    "Based on the provided Union Bank & RBI regulatory guidelines:\n\n"
                    "1. The bank must enforce strict Customer Due Diligence (CDD) procedures for high-risk accounts. "
                    "[Source: Master Direction - KYC | Regulator: Reserve Bank of India (RBI) | Section: Section 12 | Page: 4]\n"
                    "2. Suspicious Transaction Reports (STR) must be filed with FIU-IND within 7 working days. "
                    "[Source: PMLA Guidelines | Regulator: Financial Intelligence Unit - India (FIU-IND) | Section: Section 7 | Page: 2]"
                )
                return self._check_and_redact_canary_tokens(mock_text, canary_token=canary_token, system_prompt=system_prompt)

            # Format messages for Qwen chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)

            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            response_text = self._check_and_redact_canary_tokens(response_text, canary_token=canary_token, system_prompt=system_prompt)

            logger.info("Successfully generated Banking SLM response.")
            return response_text

        except Exception as e:
            logger.error(f"Error during LLM generation: {str(e)}")
            raise LLMGenerationError(f"Inference failure: {str(e)}")

    def _check_and_redact_canary_tokens(
        self,
        response_text: str,
        canary_token: Optional[str] = None,
        system_prompt: str = "",
    ) -> str:
        """Inspects response_text for canary token leakage and redacts any detected tokens."""
        if not response_text:
            return response_text

        canary_tokens = set()
        if canary_token:
            canary_tokens.add(canary_token)

        if system_prompt:
            canary_tokens.update(re.findall(rf"{CANARY_TOKEN_PREFIX}-[a-f0-9]+", system_prompt))

        found_tokens_in_response = set(re.findall(rf"{CANARY_TOKEN_PREFIX}-[a-f0-9]+", response_text))
        all_detected_tokens = canary_tokens.intersection(found_tokens_in_response) or found_tokens_in_response

        if all_detected_tokens:
            logger.error(
                "CANARY TOKEN LEAK DETECTED in LLM generator output: model echoed internal "
                "tracking token in generated text. Redacting leaked token."
            )
            for token in all_detected_tokens:
                response_text = response_text.replace(token, "[REDACTED]")

        return response_text
