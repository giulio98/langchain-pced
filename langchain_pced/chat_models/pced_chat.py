"""PCED Chat Model implementation for LangChain."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_pced.llms.pced_llm import PCEDLLM


logger = logging.getLogger(__name__)


# Keys used to pass expert data through message metadata
EXPERT_DOCS_KEY = "_pced_expert_documents"
EXPERT_SCORES_KEY = "_pced_expert_scores"
EXPERT_PLACEHOLDER_KEY = "_pced_expert_placeholder_key"
ANSWER_PREFIX_KEY = "_pced_answer_prefix"


class PCEDPromptTemplate(RunnableSerializable):
    """
    A prompt template wrapper for PCED that automatically handles expert data passing.
    
    This allows users to pass expert_documents and expert_scores in the invoke() input dict,
    and the template will pass them through to PCEDChatModel via message metadata.
    
    Example:
        ```python
        from langchain_pced import PCEDLLM, PCEDChatModel, PCEDPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = PCEDLLM.from_model_id("meta-llama/Llama-3.1-8B-Instruct")
        chat = PCEDChatModel(llm=llm)
        
        # Use normal {context} - no need for {{context}}!
        prompt = PCEDPromptTemplate.from_messages(
            messages=[
                ("system", "Answer using the context provided."),
                ("human", "Context: {context}\\n\\nQuestion: {question}"),
            ],
            expert_placeholder_key="context",
        )
        
        chain = prompt | chat | StrOutputParser()
        
        # Pass everything in invoke() - no bind() needed!
        result = chain.invoke({
            "question": "What is a panda?",
            "expert_documents": ["Doc 1...", "Doc 2..."],
            "expert_scores": [0.9, 0.8],
        })
        ```
    """
    
    template: ChatPromptTemplate
    expert_placeholder_key: str
    answer_prefix: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_messages(
        cls,
        messages: Sequence[tuple],
        expert_placeholder_key: str = "context",
        answer_prefix: Optional[str] = None,
    ) -> "PCEDPromptTemplate":
        """
        Create a PCEDPromptTemplate from a list of message tuples.
        
        Args:
            messages: List of (role, content) tuples, same as ChatPromptTemplate.from_messages
            expert_placeholder_key: The placeholder that PCED will fill per expert (default: "context")
            answer_prefix: Optional string to prepend to the model's response
        
        Returns:
            PCEDPromptTemplate instance
        """
        template = ChatPromptTemplate.from_messages(messages)
        return cls(
            template=template,
            expert_placeholder_key=expert_placeholder_key,
            answer_prefix=answer_prefix,
        )
    
    def invoke(self, input: Dict[str, Any], config: Optional[Any] = None) -> List[BaseMessage]:
        """
        Invoke the template, extracting expert data and passing it through message metadata.
        
        The expert_placeholder_key is automatically set to "{key}" so that it
        survives template formatting and remains for PCED to fill.
        
        Expert data (expert_documents, expert_scores) is extracted from input and
        attached to messages via additional_kwargs for PCEDChatModel to retrieve.
        """
        # Make a copy to avoid modifying the original input
        input_copy = dict(input)
        
        # Extract expert data from input
        expert_documents = input_copy.pop("expert_documents", None)
        expert_scores = input_copy.pop("expert_scores", None)
        
        # Inject the placeholder - it becomes "{context}" after format
        if self.expert_placeholder_key not in input_copy:
            input_copy[self.expert_placeholder_key] = "{" + self.expert_placeholder_key + "}"
        
        # Format the template - returns ChatPromptValue
        prompt_value = self.template.invoke(input_copy, config)
        
        # Extract messages from ChatPromptValue
        messages = prompt_value.to_messages()
        
        # Attach expert data to the last message's additional_kwargs
        # PCEDChatModel will extract it from there
        if messages and (expert_documents is not None or expert_scores is not None):
            last_msg = messages[-1]
            # Create new message with metadata
            metadata = {
                EXPERT_DOCS_KEY: expert_documents,
                EXPERT_SCORES_KEY: expert_scores,
                EXPERT_PLACEHOLDER_KEY: self.expert_placeholder_key,
                ANSWER_PREFIX_KEY: self.answer_prefix,
            }
            
            # Update additional_kwargs on the last message
            if hasattr(last_msg, 'additional_kwargs'):
                new_kwargs = dict(last_msg.additional_kwargs) if last_msg.additional_kwargs else {}
                new_kwargs.update(metadata)
                # Create a new message with updated kwargs
                if isinstance(last_msg, HumanMessage):
                    messages[-1] = HumanMessage(
                        content=last_msg.content,
                        additional_kwargs=new_kwargs,
                    )
                elif isinstance(last_msg, SystemMessage):
                    messages[-1] = SystemMessage(
                        content=last_msg.content,
                        additional_kwargs=new_kwargs,
                    )
                elif isinstance(last_msg, AIMessage):
                    messages[-1] = AIMessage(
                        content=last_msg.content,
                        additional_kwargs=new_kwargs,
                    )
        
        return messages
    
    @property
    def input_variables(self) -> List[str]:
        """Return input variables excluding the expert placeholder."""
        return [v for v in self.template.input_variables if v != self.expert_placeholder_key]


class PCEDChatModel(BaseChatModel):
    """
    Chat model wrapper for PCED, providing a standard LangChain chat interface.
    
    This allows using PCED with ChatPromptTemplate and the standard
    `prompt | chat_model | parser` chain pattern.
    
    Expert documents and scores can be provided either:
    1. Through the input dict (recommended): Pass expert_documents and expert_scores
       in chain.invoke() when using PCEDPromptTemplate
    2. Via bind() method (legacy): Call chat.bind() before building the chain
    
    Example (recommended - no bind):
        ```python
        from langchain_pced import PCEDLLM, PCEDChatModel, PCEDPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = PCEDLLM.from_model_id("meta-llama/Llama-3.1-8B-Instruct")
        chat = PCEDChatModel(llm=llm)
        
        prompt = PCEDPromptTemplate.from_messages(
            messages=[
                ("system", "Answer using only the context provided."),
                ("human", "Context:\\n{context}\\n\\nQuestion:\\n{question}"),
            ],
            expert_placeholder_key="context",
        )
        
        chain = prompt | chat | StrOutputParser()
        
        # Pass everything in invoke() - no bind() needed!
        result = chain.invoke({
            "question": "What is panda?",
            "expert_documents": ["doc1...", "doc2...", "doc3..."],
            "expert_scores": [0.9, 0.8, 0.7],
        })
        ```
    
    Example (legacy - with bind):
        ```python
        chain = prompt | chat.bind(
            expert_documents=["doc1...", "doc2...", "doc3..."],
            expert_scores=[0.9, 0.8, 0.7],
            expert_placeholder_key="context",
        ) | StrOutputParser()
        
        result = chain.invoke({"question": "What is panda?"})
        ```
    """
    
    llm: PCEDLLM = Field(description="The underlying PCEDLLM instance")
    
    # Bound expert data (set via bind() - legacy method)
    _expert_documents: Optional[List[str]] = PrivateAttr(default=None)
    _expert_scores: Optional[List[float]] = PrivateAttr(default=None)
    _expert_placeholder_key: Optional[str] = PrivateAttr(default=None)
    _answer_prefix: Optional[str] = PrivateAttr(default=None)
    
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    
    def __init__(
        self,
        llm: PCEDLLM,
        _expert_documents: Optional[List[str]] = None,
        _expert_scores: Optional[List[float]] = None,
        _expert_placeholder_key: Optional[str] = None,
        _answer_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        self._expert_documents = _expert_documents
        self._expert_scores = _expert_scores
        self._expert_placeholder_key = _expert_placeholder_key
        self._answer_prefix = _answer_prefix
    
    def bind(
        self,
        expert_documents: Optional[List[str]] = None,
        expert_scores: Optional[List[float]] = None,
        expert_placeholder_key: Optional[str] = None,
        answer_prefix: Optional[str] = None,
        **kwargs,
    ) -> "PCEDChatModel":
        """
        Bind expert documents and scores for PCED decoding (legacy method).
        
        NOTE: The recommended approach is to pass expert_documents and expert_scores
        directly in chain.invoke() when using PCEDPromptTemplate. This bind() method
        is kept for backward compatibility.
        
        Args:
            expert_documents: List of document strings, one per expert.
            expert_scores: List of relevance scores (REQUIRED). 
                Use harmonic_mean_fusion() to combine retrieval and reranker scores.
            expert_placeholder_key: The placeholder key in your template that will
                be filled with each expert document (e.g., "context").
            answer_prefix: Optional string to prepend to the model's response.
        
        Returns:
            New PCEDChatModel instance with bound expert data.
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
        
        return PCEDChatModel(
            llm=self.llm,
            _expert_documents=list(expert_documents),
            _expert_scores=list(expert_scores),
            _expert_placeholder_key=expert_placeholder_key,
            _answer_prefix=answer_prefix,
        )
    
    @property
    def _llm_type(self) -> str:
        return "pced_chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_id": self.llm.model_id,
            "beta": self.llm.beta,
            "gamma": self.llm.gamma,
        }
    
    def _extract_expert_data_from_messages(
        self, messages: List[BaseMessage]
    ) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[str], Optional[str]]:
        """
        Extract expert data from message metadata (set by PCEDPromptTemplate).
        
        Returns:
            Tuple of (expert_documents, expert_scores, expert_placeholder_key, answer_prefix)
        """
        for msg in reversed(messages):  # Check from last message first
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                kwargs = msg.additional_kwargs
                if EXPERT_DOCS_KEY in kwargs or EXPERT_SCORES_KEY in kwargs:
                    return (
                        kwargs.get(EXPERT_DOCS_KEY),
                        kwargs.get(EXPERT_SCORES_KEY),
                        kwargs.get(EXPERT_PLACEHOLDER_KEY),
                        kwargs.get(ANSWER_PREFIX_KEY),
                    )
        return None, None, None, None
    
    def _get_expert_data(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[str], List[float], str, Optional[str]]:
        """
        Get expert data from either message metadata or bound attributes.
        
        Priority:
        1. Message metadata (from PCEDPromptTemplate.invoke())
        2. Bound attributes (from .bind() method)
        
        Raises:
            ValueError: If no expert data is available from either source.
        """
        # Try to get from message metadata first (new API)
        docs, scores, placeholder_key, answer_prefix = self._extract_expert_data_from_messages(messages)
        
        if docs is not None and scores is not None and placeholder_key is not None:
            # Validate
            if len(docs) != len(scores):
                raise ValueError(
                    f"expert_documents ({len(docs)}) and expert_scores ({len(scores)}) "
                    "must have the same length."
                )
            if len(docs) == 0:
                raise ValueError("expert_documents cannot be empty.")
            return docs, scores, placeholder_key, answer_prefix
        
        # Fall back to bound attributes (legacy API)
        if self._expert_documents is not None and self._expert_scores is not None:
            if self._expert_placeholder_key is None:
                raise ValueError(
                    "expert_placeholder_key is required when using bind(). "
                    "Call bind() with expert_placeholder_key='context' or similar."
                )
            return (
                self._expert_documents,
                self._expert_scores,
                self._expert_placeholder_key,
                self._answer_prefix,
            )
        
        # No expert data available
        raise ValueError(
            "No expert data provided. You must either:\n\n"
            "1. (Recommended) Use PCEDPromptTemplate and pass expert data in invoke():\n"
            "   chain.invoke({\n"
            "       'question': '...',\n"
            "       'expert_documents': ['doc1', 'doc2', ...],\n"
            "       'expert_scores': [0.9, 0.8, ...],\n"
            "   })\n\n"
            "2. (Legacy) Use bind() before building the chain:\n"
            "   chain = prompt | chat.bind(\n"
            "       expert_documents=['doc1', 'doc2', ...],\n"
            "       expert_scores=[0.9, 0.8, ...],\n"
            "       expert_placeholder_key='context',\n"
            "   ) | parser"
        )
    
    def _get_max_context_length(self) -> int:
        """Get maximum context length from model config."""
        model_config = self.llm._model.config
        
        # Try different config attributes for max position embeddings
        max_pos = getattr(model_config, 'max_position_embeddings', None)
        if max_pos is None:
            max_pos = getattr(model_config, 'n_positions', None)
        if max_pos is None:
            max_pos = getattr(model_config, 'max_seq_len', None)
        if max_pos is None:
            max_pos = 4096  # Default fallback
        
        return max_pos
    
    def _truncate_context(self, text: str, max_tokens: int) -> str:
        """
        Truncate a context string to fit within max_tokens.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed
        
        Returns:
            Truncated text
        """
        tokenizer = self.llm._tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        logger.warning(
            f"Context truncated from {len(tokens)} to {max_tokens} tokens "
            f"to fit within model context window"
        )
        
        return truncated_text
    
    def _extract_system_content(self, messages: List[BaseMessage]) -> str:
        """Extract system message content from messages."""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return ""
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert messages to a prompt string using the model's chat template.
        
        IMPORTANT: The prompt string will contain the unfilled {expert_placeholder_key}
        which will be filled by the PCED LLM during generation.
        """
        # Build messages for the tokenizer's chat template
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Default to user role for unknown message types
                formatted_messages.append({"role": "user", "content": str(msg.content)})
        
        # Handle Gemma-style models that don't support system role
        model_name_lower = self.llm.model_id.lower()
        if "gemma" in model_name_lower:
            # Merge system message into first user message
            merged = []
            system_content = ""
            for m in formatted_messages:
                if m["role"] == "system":
                    system_content = m["content"] + "\n\n"
                else:
                    if m["role"] == "user" and system_content:
                        m["content"] = system_content + m["content"]
                        system_content = ""
                    merged.append(m)
            formatted_messages = merged
        
        # Apply chat template
        try:
            prompt = self.llm._tokenizer.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.llm.enable_thinking,
            )
        except TypeError:
            # Older transformers versions may not support enable_thinking
            prompt = self.llm._tokenizer.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        
        return prompt
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using PCED."""
        # Get expert data (from messages or bound attributes)
        expert_documents, expert_scores, expert_placeholder_key, answer_prefix = \
            self._get_expert_data(messages)
        
        # Convert messages to prompt string (keeping {expert_placeholder_key} unfilled)
        prompt = self._messages_to_prompt(messages)
        
        # Extract system content for building amateur prompt
        system_content = self._extract_system_content(messages)
        
        # Build amateur and expert prompts with context truncation
        prompts, question_ids = self._build_pced_prompts_with_truncation(
            prompt, system_content, expert_documents, expert_placeholder_key, answer_prefix
        )
        
        # Generate using PCED
        generated_text = self._generate_pced(prompts, question_ids, expert_scores)
        
        # Return as ChatResult
        message = AIMessage(content=generated_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _build_pced_prompts_with_truncation(
        self, 
        prompt: str, 
        system_content: str,
        documents: List[str],
        placeholder_key: str,
        answer_prefix: Optional[str],
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Build amateur and expert token sequences from the template with auto-truncation.
        
        CRITICAL: The amateur prompt must have EMPTY user content (no "Context:\n" text),
        exactly matching the reference implementation. This is done using the separator trick.
        
        Args:
            prompt: The formatted prompt string with {context} placeholder
            system_content: The system message content (extracted from messages)
            documents: List of document strings for experts
            placeholder_key: The placeholder key (e.g., "context")
            answer_prefix: Optional string to prepend to the model's response
        
        Returns:
            Tuple of (context_token_lists, suffix_ids):
            - context_token_lists: List of token ID lists for prefill
            - suffix_ids: Token IDs for the suffix (question), processed during decode
        """
        tokenizer = self.llm._tokenizer
        
        placeholder = "{" + placeholder_key + "}"
        
        if placeholder not in prompt:
            raise ValueError(
                f"Placeholder '{placeholder}' not found in the formatted prompt.\n"
                f"Make sure your ChatPromptTemplate uses '{placeholder}' "
                f"and you've specified expert_placeholder_key='{placeholder_key}'\n\n"
                f"Prompt preview: {prompt[:500]}..."
            )
        
        # Split at the placeholder to get prefix and suffix
        parts = prompt.split(placeholder, 1)
        if len(parts) != 2:
            raise ValueError(f"Could not split template at placeholder {placeholder}")
        
        prefix, suffix = parts
        
        # Calculate maximum tokens available for context
        max_context_len = self._get_max_context_length()
        max_new_tokens = self.llm.max_new_tokens
        
        # ========== BUILD AMATEUR PROMPT (EMPTY USER CONTENT) ==========
        # Use the separator trick from reference implementation
        # Amateur should have NO "Context:\n" text - just empty user content
        text = ""
        # NOTE: Don't include newlines in separator - chat templates may strip/modify them
        separator = "<<<PCED_AMATEUR_SEPARATOR_7f3a9b2c>>>"
        
        # Handle Gemma-style models
        model_name_lower = self.llm.model_id.lower()
        if "gemma" in model_name_lower:
            messages = [{"role": "user", "content": system_content + "\n\n" + text + separator}]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": text + separator}
            ]
        
        try:
            amateur_full = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            amateur_full = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        
        amateur_prompt, question_suffix = amateur_full.split(separator)
        amateur_ids = tokenizer.encode(amateur_prompt, add_special_tokens=False)
        
        # ========== BUILD SUFFIX (QUESTION PART) ==========
        # The suffix is "Question: X" + question_suffix + answer_prefix
        # We need to extract the question text from the original suffix
        # The original suffix starts right after {context} and includes the question
        
        # Append answer_prefix if provided
        suffix_with_prefix = suffix
        if answer_prefix:
            suffix_with_prefix = suffix + answer_prefix
        
        suffix_ids = tokenizer.encode(suffix_with_prefix, add_special_tokens=False)
        
        # ========== BUILD EXPERT PROMPTS ==========
        # For experts, use prefix + document (no suffix)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        
        # Maximum tokens for context placeholder
        max_context_tokens = max_context_len - max_new_tokens - len(prefix_ids) - len(suffix_ids) - 10
        max_context_tokens = max(max_context_tokens, 100)
        
        context_token_lists = []
        
        # Amateur prompt (empty user content)
        context_token_lists.append(amateur_ids)
        
        # Expert prompts (prefix + document)
        for doc in documents:
            doc_ids = tokenizer.encode(doc, add_special_tokens=False)
            if len(doc_ids) > max_context_tokens:
                doc_ids = doc_ids[:max_context_tokens]
                logger.warning(
                    f"Context truncated to {max_context_tokens} tokens to fit within model context window"
                )
            
            context_token_lists.append(prefix_ids + doc_ids)
        
        return context_token_lists, list(suffix_ids)
    
    def _generate_pced(
        self, 
        contexts_input_ids: List[List[int]], 
        question_ids: List[int],
        expert_scores: List[float],
    ) -> str:
        """
        Run PCED decoding on the token sequences.
        
        Args:
            contexts_input_ids: List of token ID lists (prefix + context).
                                First is amateur (empty context), rest are experts.
            question_ids: Token IDs for the suffix/question, processed during decode.
            expert_scores: List of relevance scores for experts.
        """
        from langchain_pced.core.prefill_cb import (
            prefill_cb_with_fork,
            compute_common_prefix_length,
        )
        from langchain_pced.core.decode_cb import decode_from_cache_cb
        from transformers import GenerationConfig
        
        # Convert scores to tensor
        scores = torch.tensor(
            expert_scores,
            device=self.llm.device,
            dtype=torch.float32,
        )
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.llm.max_new_tokens,
            eos_token_id=self.llm._tokenizer.eos_token_id,
            do_sample=self.llm.do_sample,
            temperature=self.llm.temperature,
            top_k=self.llm.top_k,
            top_p=self.llm.top_p,
        )
        
        # Validate that all prompts have tokens
        for i, ids in enumerate(contexts_input_ids):
            if len(ids) == 0:
                raise ValueError(f"Token sequence {i} is empty!")
        
        # Find common prefix length for KV cache sharing
        common_prefix_len = compute_common_prefix_length(contexts_input_ids)
        
        shared_prefix_ids = contexts_input_ids[0][:common_prefix_len]
        suffix_ids_list = [tokens[common_prefix_len:] for tokens in contexts_input_ids]
        
        # Use forking for memory-efficient KV cache sharing
        cache, past_lens, states = prefill_cb_with_fork(
            self.llm._model,
            gen_config,
            shared_prefix_ids,
            suffix_ids_list,
            len(question_ids),  # Reserve space for question tokens
        )
        
        # Decode using PCED
        generated_ids = decode_from_cache_cb(
            model=self.llm._model,
            tokenizer=self.llm._tokenizer,
            cache=cache,
            generation_config=gen_config,
            question_ids=question_ids,
            past_lens=past_lens,
            states=states,
            beta=self.llm.beta,
            retrieval_scores=scores,
            gamma=self.llm.gamma,
        )
        
        # Decode to text
        generated_text = self.llm._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Cleanup
        del cache
        gc.collect()
        if "cuda" in self.llm.device:
            torch.cuda.empty_cache()
        
        return generated_text
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream generation token by token."""
        # Get expert data (from messages or bound attributes)
        expert_documents, expert_scores, expert_placeholder_key, answer_prefix = \
            self._get_expert_data(messages)
        
        from langchain_pced.core.prefill_cb import (
            prefill_cb_with_fork,
            compute_common_prefix_length,
        )
        from langchain_pced.core.decode_cb import decode_from_cache_cb_streaming
        from transformers import GenerationConfig
        
        # Convert messages to prompt string
        prompt = self._messages_to_prompt(messages)
        
        # Extract system content for building amateur prompt
        system_content = self._extract_system_content(messages)
        
        # Build amateur and expert token sequences with truncation
        contexts_input_ids, question_ids = self._build_pced_prompts_with_truncation(
            prompt, system_content, expert_documents, expert_placeholder_key, answer_prefix
        )
        
        # Convert scores to tensor
        scores = torch.tensor(
            expert_scores,
            device=self.llm.device,
            dtype=torch.float32,
        )
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.llm.max_new_tokens,
            eos_token_id=self.llm._tokenizer.eos_token_id,
            do_sample=self.llm.do_sample,
            temperature=self.llm.temperature,
            top_k=self.llm.top_k,
            top_p=self.llm.top_p,
        )
        
        # Find common prefix length
        common_prefix_len = compute_common_prefix_length(contexts_input_ids)
        
        shared_prefix_ids = contexts_input_ids[0][:common_prefix_len]
        suffix_ids_list = [tokens[common_prefix_len:] for tokens in contexts_input_ids]
        
        # Prefill (question_ids will be processed in decode)
        cache, past_lens, states = prefill_cb_with_fork(
            self.llm._model,
            gen_config,
            shared_prefix_ids,
            suffix_ids_list,
            len(question_ids),  # Reserve space for question tokens
        )
        
        # Stream decode
        try:
            for token_str in decode_from_cache_cb_streaming(
                model=self.llm._model,
                tokenizer=self.llm._tokenizer,
                cache=cache,
                generation_config=gen_config,
                question_ids=question_ids,
                past_lens=past_lens,
                states=states,
                beta=self.llm.beta,
                retrieval_scores=scores,
                gamma=self.llm.gamma,
            ):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token_str)
                )
                if run_manager:
                    run_manager.on_llm_new_token(token_str, chunk=chunk)
                yield chunk
        finally:
            # Cleanup
            del cache
            gc.collect()
            if "cuda" in self.llm.device:
                torch.cuda.empty_cache()