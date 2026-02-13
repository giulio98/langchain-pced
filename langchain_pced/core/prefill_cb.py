"""
Continuous batching prefill logic for PCED using PagedAttention (transformers v5+).

This module provides memory-efficient KV cache management using PagedAttention,
which is available in transformers v5 and later.

Key optimization: True block sharing allows multiple experts to share KV cache blocks
for common prefixes (e.g., system prompts), significantly reducing memory usage.
"""

import math
import torch
from typing import Any, List, Tuple


def _check_continuous_batching_available() -> bool:
    """Check if continuous batching is available in the installed transformers version."""
    try:
        from transformers.generation.continuous_batching.cache import PagedAttentionCache
        from transformers.generation.continuous_batching.requests import RequestState
        return True
    except ImportError:
        return False


def compute_common_prefix_length(token_lists: List[List[int]]) -> int:
    """
    Compute the length of the longest common prefix among all token lists.
    
    Args:
        token_lists: List of token ID lists to compare
        
    Returns:
        Length of the longest common prefix shared by ALL lists
    """
    if not token_lists:
        return 0
    if len(token_lists) == 1:
        return len(token_lists[0])
    
    min_len = min(len(tl) for tl in token_lists)
    common_len = 0
    for i in range(min_len):
        first_token = token_lists[0][i]
        if all(tl[i] == first_token for tl in token_lists[1:]):
            common_len += 1
        else:
            break
    
    return common_len


class PCEDOutOfMemoryError(RuntimeError):
    """Custom exception for PCED out-of-memory conditions with helpful suggestions."""
    pass


def ensure_capacity(
    cache: Any,
    state: Any,
    total_tokens_needed: int,
    num_experts: int = 0,
) -> None:
    """
    Ensure the request has enough blocks to store KV for `total_tokens_needed` tokens.
    Allocates only the *additional* blocks required.
    
    Args:
        cache: PagedAttentionCache instance
        state: RequestState instance
        total_tokens_needed: Total number of tokens that need to fit in the cache
        num_experts: Number of experts (for error messages)
    
    Raises:
        PCEDOutOfMemoryError: If cache blocks cannot be allocated, with suggestions
    """
    bs = cache.block_size
    required_blocks = math.ceil(total_tokens_needed / bs) + 1
    to_allocate = required_blocks - state.allocated_blocks
    if to_allocate <= 0:
        return

    try:
        allocated = cache.allocate_blocks(to_allocate, state.request_id, state.allocated_blocks)
    except TypeError:
        allocated = cache.allocate_blocks(to_allocate, state)
    
    if allocated is None:
        expert_info = f" for {num_experts} experts" if num_experts > 0 else ""
        raise PCEDOutOfMemoryError(
            f"PCED Out of Memory: Could not allocate {to_allocate} cache blocks{expert_info}.\n\n"
            f"Suggestions to reduce memory usage:\n"
            f"  1. Reduce the number of experts/documents (use `num_experts` parameter to limit the number of experts)\n"
            f"  2. Reduce `max_new_tokens` to generate shorter responses\n"
            f"  3. Use a smaller model or lower precision (e.g., torch_dtype='float16')\n"
            f"  4. Free GPU memory by clearing other processes\n"
        )
    state.allocated_blocks += allocated


def build_packed_batch(cache: Any, requests: List[dict]) -> dict:
    """
    Build a packed batch for continuous batching.
    
    Args:
        cache: PagedAttentionCache instance
        requests: List of dicts with:
            - request_id: str
            - tokens: list[int] (tokens to process this step)
            - past_len: int (how many tokens are already in KV)
    
    Returns:
        Dict of kwargs to pass to model(**kwargs)
    """
    device = cache.device
    t_int = dict(device=device, dtype=torch.int32)

    input_ids = []
    position_ids = []
    cu_q = [0]
    max_seqlen_q = 0
    cu_k = [0]
    max_seqlen_k = 0
    read_index = [[] for _ in range(cache.num_groups)]
    write_index = [[] for _ in range(cache.num_groups)]

    for r in requests:
        rid = r["request_id"]
        toks = r["tokens"]
        past_len = r["past_len"]
        q_len = len(toks)
        k_len = past_len + q_len

        input_ids.extend(toks)
        position_ids.extend(range(past_len, past_len + q_len))

        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)
        max_seqlen_q = max(max_seqlen_q, q_len)
        max_seqlen_k = max(max_seqlen_k, k_len)

        if hasattr(cache, 'extend_read_and_write_indices'):
            cache.extend_read_and_write_indices(rid, past_len, q_len, read_index, write_index)
        else:
            cache.extend_read_indices(rid, past_len, q_len, read_index)
            cache.extend_write_indices(rid, past_len, q_len, write_index)

    input_ids = torch.tensor([input_ids], **t_int)
    position_ids = torch.tensor([position_ids], **t_int)
    cu_seq_lens_q = torch.tensor(cu_q, **t_int)
    cu_seq_lens_k = torch.tensor(cu_k, **t_int)
    read_index_t = [torch.tensor(x, **t_int) for x in read_index]
    write_index_t = [torch.tensor(x, **t_int) for x in write_index]

    logits_indices = []
    for i in range(len(requests)):
        logits_indices.append(cu_q[i + 1] - 1)
    logits_indices = torch.tensor(logits_indices, **t_int)

    return dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        cu_seq_lens_q=cu_seq_lens_q,
        max_seqlen_q=max_seqlen_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_seqlen_k=max_seqlen_k,
        read_index=read_index_t,
        write_index=write_index_t,
        logits_indices=logits_indices,
        cache=cache,
        use_cache=False,
    )


@torch.no_grad()
def prefill_cb_simple(
    model: Any,
    generation_config: Any,
    contexts_input_ids: List[List[int]],
    question_len: int = 0,
) -> Tuple[Any, List[int], List[Any]]:
    """
    Simple prefill WITHOUT block sharing/forking - matches reference implementation exactly.
    
    This is the standard prefill approach that processes each request independently
    without any KV cache block sharing. It's less memory efficient than prefill_cb_with_fork
    but produces identical outputs to the reference implementation.
    
    Args:
        model: HuggingFace model
        generation_config: Generation config with max_new_tokens etc.
        contexts_input_ids: List of token ID lists, one per request (amateur + experts)
        question_len: Length of question tokens (for capacity planning)
    
    Returns:
        Tuple of (cache, past_lens, states)
    """
    if not _check_continuous_batching_available():
        raise ImportError(
            "Continuous batching requires transformers v5+. "
            "Please upgrade: pip install transformers>=5.0.0"
        )
    
    from transformers.generation.continuous_batching.cache import PagedAttentionCache
    from transformers.generation.continuous_batching.requests import RequestState
    
    cache = PagedAttentionCache(
        model.config, 
        generation_config, 
        model.device, 
        model.dtype,
        allow_block_sharing=False  # Disable block sharing for exact match
    )
    
    requests = []
    past_lens = []
    states = []
    
    max_new_tokens = int(getattr(generation_config, "max_new_tokens", 0) or 0)
    
    for i, context_input_ids in enumerate(contexts_input_ids):
        request_id = "amateur" if i == 0 else f"expert_{i}"
        
        state = RequestState(
            request_id=request_id,
            initial_tokens=context_input_ids,
            tokens_to_process=context_input_ids,
            eos_token_id=generation_config.eos_token_id,
        )
        
        # Allocate enough for: context + full question + all future decode tokens
        total_needed = len(context_input_ids) + int(question_len) + max_new_tokens
        ensure_capacity(cache, state, total_tokens_needed=total_needed, num_experts=len(contexts_input_ids))
        
        requests.append({"request_id": request_id, "tokens": context_input_ids, "past_len": 0})
        past_lens.append(len(context_input_ids))
        states.append(state)
    
    prefill_batch = build_packed_batch(cache, requests=requests)
    with torch.inference_mode():
        _ = model.model(**prefill_batch)
    
    return cache, past_lens, states


@torch.no_grad()
def prefill_cb_with_fork(
    model: Any,
    generation_config: Any,
    shared_prefix_ids: List[int],
    suffix_ids_list: List[List[int]],
    question_len: int = 0,
) -> Tuple[Any, List[int], List[Any]]:
    """
    Prefill using TRUE forking for improved concurrency and memory efficiency.
    
    This implementation uses the fork_request API for efficient block sharing:
    1. Prefill shared prefix with a source request
    2. Mark complete blocks as shareable  
    3. Fork source to ALL children in ONE operation (true block sharing)
    4. Batch prefill all suffixes concurrently
    
    Key benefits:
    - Complete prefix blocks are shared by reference (ref_count++) not copied
    - Single fork operation for all children
    - All suffix prefills happen in a single concurrent batch
    
    NOTE: If shared_prefix_ids is empty (no common prefix), this function falls back
    to simple prefill without forking.
    
    Args:
        model: HuggingFace model
        generation_config: Generation config with max_new_tokens etc.
        shared_prefix_ids: Token IDs for the shared prefix (e.g., system prompt)
        suffix_ids_list: List of token ID lists for each expert's suffix (context)
        question_len: Length of question tokens (for capacity planning)
    
    Returns:
        Tuple of (cache, past_lens, states)
    """
    if not _check_continuous_batching_available():
        raise ImportError(
            "Continuous batching requires transformers v5+. "
            "Please upgrade: pip install transformers>=5.0.0"
        )
    
    from transformers.generation.continuous_batching.cache import PagedAttentionCache
    from transformers.generation.continuous_batching.requests import RequestState
    
    # Handle edge case: no shared prefix (fall back to simple prefill)
    if len(shared_prefix_ids) == 0:
        # Reconstruct full token lists and use simple prefill
        contexts_input_ids = [list(suffix) for suffix in suffix_ids_list]
        return prefill_cb_simple(model, generation_config, contexts_input_ids, question_len)
    
    cache = PagedAttentionCache(
        model.config, 
        generation_config, 
        model.device, 
        model.dtype,
        allow_block_sharing=True
    )
    
    max_new_tokens = int(getattr(generation_config, "max_new_tokens", 0) or 0)
    block_size = cache.block_size
    prefix_len = len(shared_prefix_ids)
    num_experts = len(suffix_ids_list)
    
    complete_prefix_blocks = prefix_len // block_size
    
    # PHASE 1: Prefill shared prefix with source request
    source_request_id = "_prefix_source"
    source_tokens = list(shared_prefix_ids)
    
    blocks_needed = (len(source_tokens) // block_size) + 2
    cache.allocate_blocks(blocks_needed, source_request_id, 0)
    
    source_batch = build_packed_batch(cache, requests=[{
        "request_id": source_request_id,
        "tokens": source_tokens,
        "past_len": 0
    }])
    
    with torch.inference_mode():
        _ = model.model(**source_batch)
    
    # PHASE 2: Mark complete blocks as shareable
    source_state = RequestState(
        request_id=source_request_id,
        initial_tokens=source_tokens,
        tokens_to_process=[],
        eos_token_id=generation_config.eos_token_id,
    )
    source_state.allocated_blocks = blocks_needed
    source_state.generated_tokens = []
    
    if complete_prefix_blocks > 0:
        cache.blocks_to_complete[source_request_id] = complete_prefix_blocks
        cache.mark_shareable_blocks_as_complete(source_state)
    
    # PHASE 3: TRUE forking - fork source to all children in ONE operation
    expert_ids = ["amateur"] + [f"expert_{i}" for i in range(1, num_experts)]
    
    copy_src, copy_dst = cache.fork_request(source_request_id, expert_ids)
    
    if copy_src:
        cache.copy_cache(copy_src, copy_dst)
    
    cache.free_blocks(source_request_id)
    
    # PHASE 4: Batch prefill all suffixes concurrently
    states = []
    past_lens = []
    suffix_requests = []
    
    for i, (expert_id, suffix) in enumerate(zip(expert_ids, suffix_ids_list)):
        full_tokens = list(shared_prefix_ids) + list(suffix)
        full_len = len(full_tokens)
        
        state = RequestState(
            request_id=expert_id,
            initial_tokens=full_tokens,
            tokens_to_process=[],
            eos_token_id=generation_config.eos_token_id,
        )
        
        forked_blocks = cache.group_cache_managers[0].block_table.get(expert_id, [])
        state.allocated_blocks = len(forked_blocks)
        
        total_needed = full_len + int(question_len) + max_new_tokens
        ensure_capacity(cache, state, total_tokens_needed=total_needed, num_experts=num_experts)
        
        if suffix:
            suffix_requests.append({
                "request_id": expert_id,
                "tokens": list(suffix),
                "past_len": prefix_len
            })
        
        states.append(state)
        past_lens.append(full_len)
    
    # Only run suffix prefill if there are requests with non-empty suffixes
    # This handles the case where all suffixes are empty (amateur has same prefix as experts)
    if suffix_requests:
        suffix_batch = build_packed_batch(cache, requests=suffix_requests)
        with torch.inference_mode():
            _ = model.model(**suffix_batch)
    # If suffix_requests is empty, all KV is already cached from the shared prefix
    
    return cache, past_lens, states

