"""
Pydantic data models for WaqfBackend library.

These models represent the core data structures used throughout the library:
- Ayah: A single verse from the Quran
- Segment: A transcribed audio segment
- Surah: Surah metadata
- AlignmentResult: Result of aligning a segment to an ayah
"""

from waqf_backend.models.ayah import Ayah
from waqf_backend.models.result import AlignmentResult
from waqf_backend.models.segment import Segment, SegmentType, WordTimestamp
from waqf_backend.models.surah import Surah

__all__ = [
    "Ayah",
    "Segment",
    "SegmentType",
    "WordTimestamp",
    "Surah",
    "AlignmentResult",
]
