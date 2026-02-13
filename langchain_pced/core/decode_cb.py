"""
Continuous batching decoding logic for PCED using PagedAttention (transformers v5+).

This module provides memory-efficient decoding using PagedAttention,
which is available in transformers v5 and later.

Uses HuggingFace LogitsProcessorList for proper temperature, top-k, and top-p handling,
consistent with the transformers library's generation pipeline.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain_pced.core.prefill_cb import build_packed_batch, _check_continuous_batching_available


logger = logging.getLogger(__name__)


def get_logits_warper_list(generation_config: Any) -> Optional[Any]:
    """
    Create a LogitsProcessorList with the appropriate warpers based on generation config.
    
    This uses the standard HuggingFace logits warpers:
    - TemperatureLogitsWarper: scales logits by 1/temperature
    - TopKLogitsWarper: keeps only top-k logits
    - TopPLogitsWarper: nucleus sampling, keeps smallest set summing to p
    
    The order of application follows HuggingFace's convention:
    1. Temperature (applied first to scale logits)
    2. Top-k (filter to k highest)
    3. Top-p (nucleus sampling)
    
    Args:
        generation_config: GenerationConfig with temperature, top_k, top_p settings
    
    Returns:
        LogitsProcessorList with configured warpers, or None if no warping needed
    """
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    
    warpers = LogitsProcessorList()
    
    # Get parameters from generation config
    temperature = getattr(generation_config, 'temperature', None)
    top_k = getattr(generation_config, 'top_k', None)
    top_p = getattr(generation_config, 'top_p', None)
    
    # Temperature warper - only add if temperature != 1.0 and temperature > 0
    if temperature is not None and temperature != 1.0 and temperature > 0:
        warpers.append(TemperatureLogitsWarper(temperature))
    
    # Top-k warper - only add if top_k > 0
    if top_k is not None and top_k > 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, filter_value=float('-inf')))
    
    # Top-p warper - only add if top_p < 1.0
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, filter_value=float('-inf')))
    
    return warpers if len(warpers) > 0 else None


def apply_logits_warpers(
    logits: torch.Tensor,
    warpers: Optional[Any],
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply logits warpers to a batch of logits.
    
    LogitsProcessorList expects:
    - input_ids: shape (batch_size, seq_len) - used by some processors for context
    - scores: shape (batch_size, vocab_size) - the logits to process
    
    Args:
        logits: Input logits tensor, shape (batch_size, vocab_size)
        warpers: LogitsProcessorList with configured warpers
        input_ids: Optional input token IDs (some warpers use this for context)
    
    Returns:
        Warped logits tensor with same shape as input
    """
    if warpers is None or len(warpers) == 0:
        return logits
    
    # LogitsProcessorList expects input_ids for context, but for PCED we typically
    # don't have the full history. Create a dummy tensor if not provided.
    if input_ids is None:
        # Create a minimal dummy input_ids - just needs correct batch size
        batch_size = logits.size(0)
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=logits.device)
    
    # Apply all warpers in sequence
    return warpers(input_ids, logits)


def get_jsd(p_logits: torch.Tensor, q_logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between two sets of logits.
    
    Args:
        p_logits: Logits from first distribution (amateur), shape (1, V) or (V,)
        q_logits: Logits from second distribution (experts), shape (N, V)
        eps: Small epsilon for numerical stability
    
    Returns:
        JSD values
    """
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    p = p.reshape(-1, p.size(-1)).clamp_min(eps)
    q = q.reshape(-1, q.size(-1)).clamp_min(eps)

    m = (0.5 * (p + q)).clamp_min(eps).log()

    return 0.5 * (
        F.kl_div(m, p, reduction="none", log_target=False) +
        F.kl_div(m, q, reduction="none", log_target=False)
    )


@torch.no_grad()
def decode_from_cache_cb(
    model: Any,
    tokenizer: Any,
    cache: Any,
    generation_config: Any,
    question_ids: List[int],
    past_lens: List[int],
    states: List[Any],
    beta: Optional[float] = None,
    retrieval_scores: Optional[torch.Tensor] = None,
    normalize: bool = False,
    return_winner_trace: bool = False,
    gamma: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
    """
    Decode tokens using PCED with continuous batching (PagedAttention).
    
    This is more memory efficient than the standard DynamicCache approach,
    especially for many parallel experts.
    
    Uses HuggingFace LogitsProcessorList for temperature/top-k/top-p warping,
    applied AFTER contrastive decoding fusion.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        cache: PagedAttentionCache instance from prefill_cb
        generation_config: Generation config with temperature, top_k, top_p, do_sample
        question_ids: List of question token IDs (can be empty if prompt is fully built)
        past_lens: List of past lengths for each request
        states: List of RequestState objects
        beta: Contrastive weight (None = dynamic JSD)
        retrieval_scores: Relevance scores for experts
        normalize: Whether to normalize retrieval scores
        return_winner_trace: Whether to return winning expert indices
        gamma: Weight for retrieval bias
    
    Returns:
        Generated token IDs tensor
        Optionally: Tuple of (token IDs, winner trace list)
    """
    if not _check_continuous_batching_available():
        raise ImportError(
            "Continuous batching requires transformers v5+. "
            "Please upgrade: pip install transformers>=5.0.0"
        )
    
    # Ensure question_ids is a Python list[int]
    if isinstance(question_ids, torch.Tensor):
        question_ids = question_ids.flatten().tolist()
    else:
        question_ids = list(question_ids)

    winner_trace = []
    generated_ids = []

    max_new_tokens = generation_config.max_new_tokens
    do_sample = generation_config.do_sample

    eos_ids = generation_config.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_tensor = torch.tensor(eos_ids, device=model.device, dtype=torch.long)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = int(eos_tensor[0].item())

    unfinished = torch.ones(1, dtype=torch.bool, device=model.device)

    # Compute retrieval bias
    retrieval_bias = None
    if retrieval_scores is not None:
        if normalize:
            w = torch.softmax(retrieval_scores, dim=0)
            retrieval_bias = w.clamp_min(1e-12).log()
        else:
            retrieval_bias = retrieval_scores.clamp_min(1e-12).log()

    # Step input: first consume the question tokens, then 1 token per step
    # question_ids should always have at least 1 token (guaranteed by caller)
    step_tokens = question_ids
    past_lens = list(past_lens)  # Make mutable copy
    
    # Track dynamic beta - computed once on first token if beta=None
    current_beta = beta
    
    # Create logits warpers from generation config (temperature, top-k, top-p)
    # These are applied AFTER contrastive decoding fusion
    logits_warpers = get_logits_warper_list(generation_config)

    for _ in range(max_new_tokens):
        step_requests = []
        # Build one entry per request (amateur + experts)
        for i, past_len in enumerate(past_lens):
            request_id = "amateur" if i == 0 else f"expert_{i}"
            step_requests.append({
                "request_id": request_id,
                "tokens": step_tokens,
                "past_len": past_len
            })

        decode_batch = build_packed_batch(cache, requests=step_requests)
        keep = decode_batch["logits_indices"].to(torch.long)
        
        with torch.inference_mode():
            out = model(**decode_batch, logits_to_keep=keep)
        
        next_logits = out.logits[0]  # [num_requests, V]

        amateur_logits = next_logits[0].unsqueeze(0)  # [1, V]
        expert_logits = next_logits[1:]               # [N, V]

        # Compute dynamic beta ONCE on first token (when beta=None and current_beta is None)
        if current_beta is None:
            current_beta = get_jsd(amateur_logits, expert_logits).clamp(0, 1.0)

        # Apply contrastive decoding formula
        calibrated_scores = (1 + current_beta) * expert_logits - current_beta * amateur_logits

        # Apply logits warpers (temperature, top-k, top-p) AFTER contrastive fusion
        # This follows HuggingFace's standard generation pipeline
        calibrated_scores = apply_logits_warpers(calibrated_scores, logits_warpers)

        # Select next token
        if do_sample:
            probs = torch.softmax(calibrated_scores, dim=-1)
            next_tokens_per_expert = torch.multinomial(probs, num_samples=1).squeeze(1)
            best_scores_per_expert = torch.gather(
                calibrated_scores, 1, next_tokens_per_expert.unsqueeze(1)
            ).squeeze(1)
            best_tokens_per_expert = next_tokens_per_expert
        else:
            best_scores_per_expert, best_tokens_per_expert = calibrated_scores.max(dim=-1)

        # Apply retrieval bias to scores (not logits) for winner selection
        if retrieval_bias is not None:
            best_scores_per_expert = best_scores_per_expert + gamma * retrieval_bias

        winning_expert_idx = int(best_scores_per_expert.argmax().item())
        if return_winner_trace:
            winner_trace.append(winning_expert_idx)

        next_token = best_tokens_per_expert[winning_expert_idx]

        # Check for EOS
        finished_now = torch.isin(next_token, eos_tensor)
        next_token = torch.where(
            unfinished, next_token, torch.full_like(next_token, pad_token_id)
        )

        generated_ids.append(int(next_token.item()))
        unfinished = unfinished & ~finished_now
        
        if not torch.any(unfinished):
            break

        # Update past_lens for all requests
        q_len = len(step_tokens)
        for i in range(len(past_lens)):
            past_lens[i] += q_len

        # Next step processes just the single new token
        step_tokens = [int(next_token.item())]

    out_tokens = torch.tensor(generated_ids, dtype=torch.long, device=model.device)
    
    if return_winner_trace:
        return out_tokens, winner_trace
    return out_tokens


@torch.no_grad()
def decode_from_cache_cb_streaming(
    model: Any,
    tokenizer: Any,
    cache: Any,
    generation_config: Any,
    question_ids: List[int],
    past_lens: List[int],
    states: List[Any],
    beta: Optional[float] = None,
    retrieval_scores: Optional[torch.Tensor] = None,
    normalize: bool = False,
    gamma: float = 1.0,
) -> Iterator[str]:
    """
    Streaming version of decode_from_cache_cb that yields tokens as they're generated.
    
    Uses HuggingFace LogitsProcessorList for temperature/top-k/top-p warping,
    applied AFTER contrastive decoding fusion.
    
    Args:
        Same as decode_from_cache_cb, except no return_winner_trace
    
    Yields:
        Token strings as they are generated
    """
    if not _check_continuous_batching_available():
        raise ImportError(
            "Continuous batching requires transformers v5+. "
            "Please upgrade: pip install transformers>=5.0.0"
        )
    
    if isinstance(question_ids, torch.Tensor):
        question_ids = question_ids.flatten().tolist()
    else:
        question_ids = list(question_ids)

    max_new_tokens = generation_config.max_new_tokens
    do_sample = generation_config.do_sample

    eos_ids = generation_config.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]

    retrieval_bias = None
    if retrieval_scores is not None:
        if normalize:
            w = torch.softmax(retrieval_scores, dim=0)
            retrieval_bias = w.clamp_min(1e-12).log()
        else:
            retrieval_bias = retrieval_scores.clamp_min(1e-12).log()

    step_tokens = question_ids
    past_lens = list(past_lens)
    
    # Track dynamic beta - computed once on first token if beta=None
    current_beta = beta
    
    # Create logits warpers from generation config (temperature, top-k, top-p)
    # These are applied AFTER contrastive decoding fusion
    logits_warpers = get_logits_warper_list(generation_config)

    for _ in range(max_new_tokens):
        step_requests = []
        for i, past_len in enumerate(past_lens):
            request_id = "amateur" if i == 0 else f"expert_{i}"
            step_requests.append({
                "request_id": request_id,
                "tokens": step_tokens,
                "past_len": past_len
            })

        decode_batch = build_packed_batch(cache, requests=step_requests)
        keep = decode_batch["logits_indices"].to(torch.long)
        
        with torch.inference_mode():
            out = model(**decode_batch, logits_to_keep=keep)
        
        next_logits = out.logits[0]
        amateur_logits = next_logits[0].unsqueeze(0)
        expert_logits = next_logits[1:]

        # Compute dynamic beta ONCE on first token (when beta=None and current_beta is None)
        if current_beta is None:
            current_beta = get_jsd(amateur_logits, expert_logits).clamp(0, 1.0)

        calibrated_scores = (1 + current_beta) * expert_logits - current_beta * amateur_logits

        # Apply logits warpers (temperature, top-k, top-p) AFTER contrastive fusion
        calibrated_scores = apply_logits_warpers(calibrated_scores, logits_warpers)

        if do_sample:
            probs = torch.softmax(calibrated_scores, dim=-1)
            next_tokens_per_expert = torch.multinomial(probs, num_samples=1).squeeze(1)
            best_scores_per_expert = torch.gather(
                calibrated_scores, 1, next_tokens_per_expert.unsqueeze(1)
            ).squeeze(1)
            best_tokens_per_expert = next_tokens_per_expert
        else:
            best_scores_per_expert, best_tokens_per_expert = calibrated_scores.max(dim=-1)

        if retrieval_bias is not None:
            best_scores_per_expert = best_scores_per_expert + gamma * retrieval_bias

        winning_expert_idx = int(best_scores_per_expert.argmax().item())
        next_token = best_tokens_per_expert[winning_expert_idx]

        # Check for EOS before yielding
        if next_token.item() in eos_ids:
            break

        # Yield the token string
        token_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
        yield token_str

        # Update for next iteration
        q_len = len(step_tokens)
        for i in range(len(past_lens)):
            past_lens[i] += q_len

        step_tokens = [int(next_token.item())]