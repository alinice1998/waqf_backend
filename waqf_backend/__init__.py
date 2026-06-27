"""
مُنَجِّم (Waqf Backend) — A Python library to synchronize Quran Ayat with audio recitations.

Usage:
    from waqf_backend.transcription import Whisperx
    from waqf_backend.core import align_segments
    from waqf_backend.data import load_surah_ayahs

    # Transcribe
    transcriber = Whisperx(model_name="model_id")
    segments = transcriber.transcribe("surah_1.wav", surah_id=1)

    # Align
    ayahs = load_surah_ayahs(1)
    results = align_segments(segments, ayahs)

    # Results contain timing information for each ayah
    for result in results:
        print(f"Ayah {result.ayah.ayah_number}: {result.start_time:.2f}s - {result.end_time:.2f}s")
"""

from waqf_backend._version import __version__
from waqf_backend.config import Waqf BackendSettings, configure, get_settings
from waqf_backend.exceptions import (
    AlignmentError,
    AudioFileError,
    ConfigurationError,
    ModelNotLoadedError,
    Waqf BackendError,
    QuranDataError,
    TranscriptionError,
)
from waqf_backend.formatters import (
    AlignmentMetadata,
    AlignmentOutput,
    FormattedAyahResult,
    format_alignment_results,
)
from waqf_backend.models import (
    AlignmentResult,
    Ayah,
    Segment,
    SegmentType,
    Surah,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Ayah",
    "Segment",
    "SegmentType",
    "Surah",
    "AlignmentResult",
    # Config
    "Waqf BackendSettings",
    "get_settings",
    "configure",
    # Formatters
    "AlignmentOutput",
    "FormattedAyahResult",
    "AlignmentMetadata",
    "format_alignment_results",
    # Exceptions
    "Waqf BackendError",
    "TranscriptionError",
    "AlignmentError",
    "ConfigurationError",
    "AudioFileError",
    "ModelNotLoadedError",
    "QuranDataError",
]
