"""Public utility functions for scoring in PCED."""

from typing import List, Union
import torch


def harmonic_mean_fusion(
    retrieval_scores: Union[List[float], torch.Tensor],
    reranker_scores: Union[List[float], torch.Tensor],
    eps: float = 1e-8,
) -> List[float]:
    """
    Fuse retrieval and reranker scores using harmonic mean.
    
    Harmonic mean is more conservative than arithmetic mean - it penalizes
    cases where one score is low even if the other is high. This is useful
    for RAG where both retrieval relevance and reranker confidence matter.
    
    Formula: 2 * (r1 * r2) / (r1 + r2 + eps)
    
    Args:
        retrieval_scores: Retrieval scores (e.g., from vector similarity).
            Should be normalized to [0, 1] range for best results.
        reranker_scores: Reranker scores (e.g., from cross-encoder).
            Should be normalized to [0, 1] range for best results.
        eps: Small epsilon to avoid division by zero.
    
    Returns:
        List of fused scores, one per document.
    
    Example:
        ```python
        from langchain_pced import harmonic_mean_fusion
        
        # Get scores from your retrieval and reranking pipeline
        vec_scores = [0.9, 0.8, 0.7]  # From vector similarity
        rerank_scores = [0.95, 0.6, 0.85]  # From cross-encoder
        
        # Fuse them
        fused = harmonic_mean_fusion(vec_scores, rerank_scores)
        # fused ≈ [0.924, 0.686, 0.768]
        ```
    """
    # Convert to tensors if needed
    if isinstance(retrieval_scores, list):
        retrieval_scores = torch.tensor(retrieval_scores, dtype=torch.float32)
    if isinstance(reranker_scores, list):
        reranker_scores = torch.tensor(reranker_scores, dtype=torch.float32)
    
    # Compute harmonic mean
    fused = 2 * (retrieval_scores * reranker_scores) / (retrieval_scores + reranker_scores + eps)
    
    return fused.tolist()