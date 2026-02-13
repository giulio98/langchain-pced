"""Internal scoring utilities for PCED decoding."""

import torch


def compute_retrieval_bias(
    scores: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute the retrieval bias term for contrastive decoding.
    
    This bias is added to the calibrated scores to weight experts
    by their retrieval relevance.
    
    Args:
        scores: Retrieval scores tensor, shape (num_experts,)
        normalize: If True, apply softmax normalization before log
    
    Returns:
        Log-space bias tensor, shape (num_experts,)
    """
    if normalize:
        # Apply softmax to convert to probabilities, then take log
        w = torch.softmax(scores, dim=0)
        bias = w.clamp_min(1e-12).log()
    else:
        # Direct log of scores (scores should already be normalized to [0,1])
        bias = scores.clamp_min(1e-12).log()
    
    return bias