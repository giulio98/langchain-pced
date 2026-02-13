"""PCED LLM implementation for LangChain."""

from __future__ import annotations

import gc
import logging
import re
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict, Field, PrivateAttr


logger = logging.getLogger(__name__)


class PCEDLLM(BaseLLM):
    """
    Parallel Context-of-Experts Decoding (PCED) LLM for LangChain.
    
    PCED treats retrieved documents as isolated "experts" that process the query
    independently. During decoding, it aggregates expert predictions via contrastive
    decoding, recovering cross-document reasoning without shared attention.
    
    This LLM requires expert documents and scores to be bound before generation.
    Use the `bind()` method to provide:
    - `expert_documents`: List of document strings (one per expert)
    - `expert_scores`: List of relevance scores (REQUIRED - use harmonic_mean_fusion)
    - `expert_placeholder_key`: The placeholder in your template to fill per expert
    
    Example:
        ```python
        from langchain_pced import PCEDLLM, harmonic_mean_fusion
        from langchain_core.prompts import ChatPromptTemplate
        
        # Create LLM
        llm = PCEDLLM.from_model_id("meta-llama/Llama-3.1-8B-Instruct")
        
        # Bind expert data (scores are REQUIRED)
        bound_llm = llm.bind(
            expert_documents=["doc1...", "doc2...", "doc3..."],
            expert_scores=[0.9, 0.8, 0.7],  # Pre-computed with harmonic_mean_fusion
            expert_placeholder_key="context",
        )
        
        # Use with a prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using the context."),
            ("human", "Context: {context}\\n\\nQuestion: {question}"),
        ])
        
        chain = prompt | bound_llm
        result = chain.invoke({"question": "What is AI?"})
        ```
    """
    
    model_id: str = Field(description="HuggingFace model ID")
    
    # PCED parameters
    beta: Optional[float] = Field(
        default=None,
        description="Contrastive weight. None = dynamic JSD-based"
    )
    gamma: float = Field(
        default=2.5,
        description="Retrieval score weight for expert selection"
    )
    
    # Generation parameters - None means use model's generation_config defaults
    max_new_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature (None = model default)")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling (None = model default)")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling (None = model default)")
    do_sample: Optional[bool] = Field(default=None, description="Whether to sample (None = model default)")
    enable_thinking: bool = Field(
        default=False,
        description="Enable thinking/reasoning mode for models like Qwen3"
    )
    
    # Device configuration
    device: str = Field(default="cuda:0", description="Device to run on")
    torch_dtype: str = Field(default="bfloat16", description="Torch dtype")
    
    # Bound expert data (set via bind())
    _expert_documents: Optional[List[str]] = PrivateAttr(default=None)
    _expert_scores: Optional[List[float]] = PrivateAttr(default=None)
    _expert_placeholder_key: Optional[str] = PrivateAttr(default=None)
    
    # Private model attributes
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **kwargs):
        # Extract private attrs before calling super
        expert_documents = kwargs.pop("_expert_documents", None)
        expert_scores = kwargs.pop("_expert_scores", None)
        expert_placeholder_key = kwargs.pop("_expert_placeholder_key", None)
        
        super().__init__(**kwargs)
        
        self._expert_documents = expert_documents
        self._expert_scores = expert_scores
        self._expert_placeholder_key = expert_placeholder_key
        
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)
        
        logger.info(f"Loading model: {self.model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Use paged attention for continuous batching
        attn_implementation = "paged|flash_attention_2"
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self._model.eval()
        logger.info(f"Model loaded on {self.device}")
        
        # Apply model generation defaults
        self._apply_model_generation_defaults()
    
    def _apply_model_generation_defaults(self):
        """Apply model's default generation parameters if not explicitly set."""
        if self._model is None:
            return
        
        model_gen_config = getattr(self._model, 'generation_config', None)
        if model_gen_config is None:
            if self.do_sample is None:
                object.__setattr__(self, 'do_sample', False)
            if self.temperature is None:
                object.__setattr__(self, 'temperature', 1.0)
            if self.top_p is None:
                object.__setattr__(self, 'top_p', 1.0)
            return
        
        if self.do_sample is None:
            object.__setattr__(self, 'do_sample', getattr(model_gen_config, 'do_sample', False))
        if self.temperature is None:
            temp = getattr(model_gen_config, 'temperature', 1.0)
            object.__setattr__(self, 'temperature', float(temp) if temp else 1.0)
        if self.top_p is None:
            top_p = getattr(model_gen_config, 'top_p', 1.0)
            object.__setattr__(self, 'top_p', float(top_p) if top_p else 1.0)
        if self.top_k is None:
            top_k = getattr(model_gen_config, 'top_k', None)
            if top_k is not None:
                object.__setattr__(self, 'top_k', int(top_k))
    
    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        beta: Optional[float] = None,
        gamma: float = 2.5,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        enable_thinking: bool = False,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> "PCEDLLM":
        """
        Create a PCEDLLM from a model ID.
        
        Args:
            model_id: HuggingFace model ID
            beta: Contrastive weight (None = dynamic JSD)
            gamma: Retrieval score weight for expert selection
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None = model default)
            top_k: Top-k sampling (None = model default)
            top_p: Top-p sampling (None = model default)
            do_sample: Whether to sample (None = model default)
            enable_thinking: Enable thinking mode for Qwen3-like models
            device: Device to run on
            torch_dtype: Torch dtype string
            
        Returns:
            PCEDLLM instance
        """
        return cls(
            model_id=model_id,
            beta=beta,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            enable_thinking=enable_thinking,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    
    def bind(
        self,
        expert_documents: Optional[List[str]] = None,
        expert_scores: Optional[List[float]] = None,
        expert_placeholder_key: Optional[str] = None,
        **kwargs,
    ) -> "PCEDLLM":
        """
        Bind expert documents and scores for PCED decoding.
        
        This method returns a new PCEDLLM instance with the bound data.
        All three parameters are REQUIRED for PCED to work.
        
        Args:
            expert_documents: List of document strings, one per expert.
            expert_scores: List of relevance scores (REQUIRED). 
                Use harmonic_mean_fusion() to combine retrieval and reranker scores.
            expert_placeholder_key: The placeholder key in your template that will
                be filled with each expert document (e.g., "context").
        
        Returns:
            New PCEDLLM instance with bound expert data.
        
        Raises:
            ValueError: If any required parameter is missing or invalid.
        
        Example:
            ```python
            llm = PCEDLLM.from_model_id("meta-llama/Llama-3.1-8B-Instruct")
            
            # Compute scores externally
            fused_scores = harmonic_mean_fusion(vec_scores, rerank_scores)
            
            # Bind expert data
            bound_llm = llm.bind(
                expert_documents=["doc1", "doc2", "doc3"],
                expert_scores=fused_scores,
                expert_placeholder_key="context",
            )
            ```
        """
        # Validate required parameters
        if expert_scores is None:
            raise ValueError(
                "expert_scores is REQUIRED. PCED requires pre-computed relevance scores.\n"
                "Use langchain_pced.harmonic_mean_fusion() to combine retrieval and reranker scores:\n\n"
                "    from langchain_pced import harmonic_mean_fusion\n"
                "    scores = harmonic_mean_fusion(retrieval_scores, reranker_scores)\n"
            )
        
        if expert_documents is None:
            raise ValueError(
                "expert_documents is REQUIRED. Provide a list of document strings, one per expert."
            )
        
        if expert_placeholder_key is None:
            raise ValueError(
                "expert_placeholder_key is REQUIRED. Specify which placeholder in your template "
                "should be filled with expert documents (e.g., 'context').\n\n"
                "Example: If your template is 'Context: {context}\\nQuestion: {question}',\n"
                "then expert_placeholder_key='context'"
            )
        
        if len(expert_documents) != len(expert_scores):
            raise ValueError(
                f"expert_documents ({len(expert_documents)}) and expert_scores ({len(expert_scores)}) "
                "must have the same length."
            )
        
        if len(expert_documents) == 0:
            raise ValueError("expert_documents cannot be empty.")
        
        # Create a new instance with bound data
        # We need to copy the model and tokenizer references
        new_instance = PCEDLLM.__new__(PCEDLLM)
        
        # Copy all fields
        for field_name in self.model_fields:
            setattr(new_instance, field_name, getattr(self, field_name))
        
        # Copy model and tokenizer references (not deep copy)
        new_instance._model = self._model
        new_instance._tokenizer = self._tokenizer
        
        # Set bound data
        new_instance._expert_documents = list(expert_documents)
        new_instance._expert_scores = list(expert_scores)
        new_instance._expert_placeholder_key = expert_placeholder_key
        
        return new_instance
    
    @property
    def _llm_type(self) -> str:
        return "pced_llm"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "beta": self.beta,
            "gamma": self.gamma,
        }
    
    def _validate_bound_data(self):
        """Validate that expert data has been bound."""
        if self._expert_documents is None:
            raise ValueError(
                "No expert documents bound. Call .bind() with expert_documents, "
                "expert_scores, and expert_placeholder_key before generation."
            )
        if self._expert_scores is None:
            raise ValueError(
                "No expert scores bound. Call .bind() with expert_scores before generation."
            )
        if self._expert_placeholder_key is None:
            raise ValueError(
                "No expert_placeholder_key bound. Call .bind() to specify which placeholder "
                "in your template should be filled per expert."
            )
    
    def _build_prompts_from_template(
        self,
        template_str: str,
    ) -> tuple:
        """
        Build amateur and expert prompts from a template string.
        
        The template contains {expert_placeholder_key} which will be:
        - Empty string for the amateur
        - Filled with each document for experts
        
        Args:
            template_str: The prompt template with {expert_placeholder_key}
        
        Returns:
            Tuple of (prompts_list, question_suffix)
            where prompts_list[0] is amateur and prompts_list[1:] are experts
        """
        placeholder_key = self._expert_placeholder_key
        documents = self._expert_documents
        
        # Verify placeholder exists in template
        placeholder_pattern = "{" + placeholder_key + "}"
        if placeholder_pattern not in template_str:
            raise ValueError(
                f"Placeholder '{{{placeholder_key}}}' not found in template.\n"
                f"Template: {template_str[:200]}..."
            )
        
        # Use a unique separator to split prefix from suffix
        separator = "\n<|PCED_SEPARATOR_UNIQUE_TOKEN|>\n"
        
        # Replace the placeholder with separator to find split point
        template_with_sep = template_str.replace(placeholder_pattern, separator, 1)
        
        if separator not in template_with_sep:
            raise ValueError(f"Failed to find placeholder {{{placeholder_key}}} in template")
        
        prefix, suffix = template_with_sep.split(separator, 1)
        
        prompts = []
        
        # Amateur prompt (empty context)
        amateur_prompt = prefix + "" + suffix
        prompts.append(amateur_prompt)
        
        # Expert prompts (with documents)
        for doc in documents:
            expert_prompt = prefix + doc + suffix
            prompts.append(expert_prompt)
        
        # The "question suffix" is what comes after all prompts are built
        # In PCED, all prompts share the same suffix (the question part)
        # We return the suffix as the shared part
        return prompts, suffix
    
    def _build_chat_prompts(
        self,
        messages_str: str,
    ) -> tuple:
        """
        Build amateur and expert prompts for chat-style input.
        
        For chat models, the input is typically already formatted by the tokenizer's
        chat template. We need to split at the expert placeholder.
        
        CRITICAL: To match the reference implementation's performance, we split at the
        placeholder so that:
        - Prefill only processes: prefix + context (different per expert)
        - Decode processes: suffix (same for all experts, processed once in batch)
        
        This ensures the question/suffix tokens are processed ONCE across all experts
        in a single batched forward pass during decode, rather than being computed
        separately for each expert during prefill.
        
        Args:
            messages_str: The formatted chat string with {expert_placeholder_key}
        
        Returns:
            Tuple of (prefix_prompts, suffix)
            - prefix_prompts: List of strings (prefix + context per expert, WITHOUT suffix)
            - suffix: The question/suffix part (to be tokenized and passed to decode)
        """
        placeholder_key = self._expert_placeholder_key
        documents = self._expert_documents
        
        placeholder = "{" + placeholder_key + "}"
        
        if placeholder not in messages_str:
            raise ValueError(
                f"Placeholder '{placeholder}' not found in the formatted prompt.\n"
                f"Make sure your ChatPromptTemplate uses '{placeholder}' "
                f"and you've specified expert_placeholder_key='{placeholder_key}'"
            )
        
        # Split at the placeholder - this is the critical split point
        # prefix = everything before {context}
        # suffix = everything after {context} (the question part)
        parts = messages_str.split(placeholder, 1)
        if len(parts) != 2:
            raise ValueError(f"Could not split template at placeholder {placeholder}")
        
        prefix, suffix = parts
        
        # Build prompts WITHOUT suffix - suffix will be passed to decode
        prompts = []
        
        # Amateur prompt (prefix only, no context, no suffix)
        prompts.append(prefix)
        
        # Expert prompts (prefix + document, no suffix)
        for doc in documents:
            prompts.append(prefix + doc)
        
        # Return prompts without suffix, and the suffix separately
        return prompts, suffix
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses using PCED."""
        self._validate_bound_data()
        
        generations = []
        for prompt in prompts:
            text = self._generate_single(prompt)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    def _generate_single(self, prompt: str) -> str:
        """Generate a single response using PCED with continuous batching."""
        from langchain_pced.core.prefill_cb import (
            prefill_cb_with_fork,
            compute_common_prefix_length,
        )
        from langchain_pced.core.decode_cb import decode_from_cache_cb
        from transformers import GenerationConfig
        
        # Build amateur and expert prompts (WITHOUT suffix)
        # prompts = prefix + context per expert
        # suffix = question part (to be processed in decode)
        prompts, suffix = self._build_chat_prompts(prompt)
        
        # Convert scores to tensor
        scores = torch.tensor(self._expert_scores, device=self.device, dtype=torch.float32)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        
        # Tokenize context prompts (prefix + document, no suffix)
        contexts_input_ids = [
            self._tokenizer(p, add_special_tokens=False)["input_ids"]
            for p in prompts
        ]
        
        # Tokenize suffix (question) - this will be processed in decode phase
        # This is critical for performance: question tokens are processed ONCE
        # across all experts in a single batch forward pass
        question_ids = self._tokenizer(suffix, add_special_tokens=False)["input_ids"]
        
        # Find common prefix length for KV cache sharing
        common_prefix_len = compute_common_prefix_length(contexts_input_ids)
        
        shared_prefix_ids = contexts_input_ids[0][:common_prefix_len]
        suffix_ids_list = [tokens[common_prefix_len:] for tokens in contexts_input_ids]
        
        # Prefill with fork-based block sharing
        cache, past_lens, states = prefill_cb_with_fork(
            self._model,
            gen_config,
            shared_prefix_ids,
            suffix_ids_list,
            len(question_ids),  # Reserve space for question tokens
        )
        
        # Decode using PCED - question_ids processed first, then generation
        generated_ids = decode_from_cache_cb(
            model=self._model,
            tokenizer=self._tokenizer,
            cache=cache,
            generation_config=gen_config,
            question_ids=question_ids,
            past_lens=past_lens,
            states=states,
            beta=self.beta,
            retrieval_scores=scores,
            gamma=self.gamma,
        )
        
        # Decode to text
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Cleanup
        del cache
        gc.collect()
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        
        return generated_text
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream generation token by token."""
        self._validate_bound_data()
        
        from langchain_pced.core.prefill_cb import (
            prefill_cb_with_fork,
            compute_common_prefix_length,
        )
        from langchain_pced.core.decode_cb import decode_from_cache_cb_streaming
        from transformers import GenerationConfig
        
        # Build amateur and expert prompts (WITHOUT suffix)
        # prompts = prefix + context per expert
        # suffix = question part (to be processed in decode)
        prompts, suffix = self._build_chat_prompts(prompt)
        
        # Convert scores to tensor
        scores = torch.tensor(self._expert_scores, device=self.device, dtype=torch.float32)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        
        # Tokenize context prompts (prefix + document, no suffix)
        contexts_input_ids = [
            self._tokenizer(p, add_special_tokens=False)["input_ids"]
            for p in prompts
        ]
        
        # Tokenize suffix (question) - this will be processed in decode phase
        question_ids = self._tokenizer(suffix, add_special_tokens=False)["input_ids"]
        
        # Find common prefix length
        common_prefix_len = compute_common_prefix_length(contexts_input_ids)
        
        shared_prefix_ids = contexts_input_ids[0][:common_prefix_len]
        suffix_ids_list = [tokens[common_prefix_len:] for tokens in contexts_input_ids]
        
        # Prefill
        cache, past_lens, states = prefill_cb_with_fork(
            self._model,
            gen_config,
            shared_prefix_ids,
            suffix_ids_list,
            len(question_ids),  # Reserve space for question tokens
        )
        
        # Stream decode
        try:
            for token_str in decode_from_cache_cb_streaming(
                model=self._model,
                tokenizer=self._tokenizer,
                cache=cache,
                generation_config=gen_config,
                question_ids=question_ids,
                past_lens=past_lens,
                states=states,
                beta=self.beta,
                retrieval_scores=scores,
                gamma=self.gamma,
            ):
                chunk = GenerationChunk(text=token_str)
                if run_manager:
                    run_manager.on_llm_new_token(token_str, chunk=chunk)
                yield chunk
        finally:
            # Cleanup
            del cache
            gc.collect()
            if "cuda" in self.device:
                torch.cuda.empty_cache()
