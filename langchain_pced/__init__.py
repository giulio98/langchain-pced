"""LangChain integration for Parallel Context-of-Experts Decoding (PCED)."""

from langchain_pced.llms.pced_llm import PCEDLLM
from langchain_pced.chat_models.pced_chat import PCEDChatModel, PCEDPromptTemplate
from langchain_pced.utils.scoring import harmonic_mean_fusion
from langchain_pced.core.prefill_cb import PCEDOutOfMemoryError

__all__ = [
    "PCEDLLM",
    "PCEDChatModel",
    "PCEDPromptTemplate",
    "harmonic_mean_fusion",
    "PCEDOutOfMemoryError",
]
__version__ = "0.2.0"