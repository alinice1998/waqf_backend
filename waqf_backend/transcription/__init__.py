"""
Transcription module for WaqfBackend library.

Provides abstract interface and implementations for audio transcription.

Note: Whisperx is imported lazily to avoid requiring the whisperx
package in environments that only need silence detection or base classes.
"""

from waqf_backend.transcription.base import BaseTranscriber
from waqf_backend.transcription.silence import (
    detect_non_silent_chunks,
    detect_silences,
    detect_silences_adaptive,
)


def __getattr__(name: str):
    """Lazy import for Whisperx to avoid requiring whisperx package at import time."""
    if name == "Whisperx":
        from waqf_backend.transcription.whisperx import Whisperx
        return Whisperx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTranscriber",
    "Whisperx",
    "detect_silences",
    "detect_silences_adaptive",
    "detect_non_silent_chunks",
]
