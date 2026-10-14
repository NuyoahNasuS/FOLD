"""FOLD: far memory with local order for block-parallel drafters."""
from .attention import FoldError, install, use_full_draft_storage

__all__ = ["FoldError", "install", "use_full_draft_storage"]
