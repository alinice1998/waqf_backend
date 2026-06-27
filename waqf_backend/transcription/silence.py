"""
Silence detection utilities for audio processing.

Provides adaptive (dynamic threshold) and fixed-threshold implementations.
The adaptive method is strongly recommended for Quranic recitations which
typically contain reverb and ambient noise that defeat fixed thresholds.
"""

from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger("waqf_backend.silence")


# ═══════════════════════════════════════════════════════════════════════════
#  PRIMARY API — Adaptive (Dynamic Threshold) Silence Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_silences_adaptive(
    audio_path: str | Path,
    min_silence_len: int = 200,
    percentile: float = 15.0,
    smooth_kernel: int = 7,
    merge_gap_ms: int = 80,
) -> list[tuple[int, int]]:
    """
    Detect silent portions using a dynamic threshold based on signal statistics.

    Instead of a fixed dB threshold (which fails on reverb-heavy recordings),
    this computes the energy percentile of the signal to automatically adapt
    to the noise floor of each specific recording.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds (default 200ms)
        percentile: Energy percentile below which frames are considered silent.
                    Lower = stricter (fewer silences detected). Default 15.0.
        smooth_kernel: Median-filter kernel size for smoothing the energy
                       contour before thresholding. Odd number. Default 7.
        merge_gap_ms: Merge silence regions separated by less than this (ms).
                      Prevents tiny speech blips from splitting one pause. Default 80.

    Returns:
        List of (start_ms, end_ms) tuples for silent portions
    """
    import librosa
    import numpy as np

    # Load audio at native sample rate for accuracy
    y, sr = librosa.load(str(audio_path), sr=None)

    # Calculate frame-based RMS energy with ~10ms frames
    frame_length = int(sr * 0.01)  # 10ms frames
    hop_length = frame_length // 2  # 50% overlap for good resolution

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # ── Adaptive Threshold ─────────────────────────────────────────────
    # Use a percentile of the RMS energy as the silence threshold.
    # This automatically adapts to the recording's noise floor.
    # For clean recordings, the threshold will be very low.
    # For reverb-heavy recordings, it will be proportionally higher.
    threshold = np.percentile(rms, percentile)

    # Ensure threshold is at least a tiny value to avoid detecting
    # everything as silence in perfectly silent recordings
    if threshold < 1e-6:
        threshold = 1e-6

    logger.debug(
        f"Adaptive silence: percentile={percentile}, "
        f"threshold={threshold:.6f}, "
        f"rms_min={rms.min():.6f}, rms_max={rms.max():.6f}, "
        f"rms_median={np.median(rms):.6f}"
    )

    # ── Smoothing ──────────────────────────────────────────────────────
    # Apply median filter to smooth out brief spikes/dips in energy.
    # This prevents false silence detections caused by momentary drops
    # (e.g., between syllables within a single word).
    if smooth_kernel > 1:
        from scipy.ndimage import median_filter
        rms_smoothed = median_filter(rms, size=smooth_kernel)
    else:
        rms_smoothed = rms

    # ── Detect silent frames ───────────────────────────────────────────
    is_silent = rms_smoothed < threshold

    # Convert frame indices to milliseconds
    frame_times_ms = (
        librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
    )

    # ── Extract contiguous silent regions ──────────────────────────────
    silences: list[tuple[int, int]] = []
    in_silence = False
    silence_start = 0.0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            silence_start = frame_times_ms[i]
        elif not silent and in_silence:
            in_silence = False
            silence_end = frame_times_ms[i]
            duration = silence_end - silence_start
            if duration >= min_silence_len:
                silences.append((int(silence_start), int(silence_end)))

    # Handle audio ending in silence
    if in_silence:
        silence_end = frame_times_ms[-1]
        duration = silence_end - silence_start
        if duration >= min_silence_len:
            silences.append((int(silence_start), int(silence_end)))

    # ── Merge nearby silences ──────────────────────────────────────────
    # Short speech blips between two silences are often noise/reverb tail.
    if len(silences) > 1 and merge_gap_ms > 0:
        merged = [silences[0]]
        for start, end in silences[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_gap_ms:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        silences = merged

    logger.info(f"Adaptive silence detection found {len(silences)} silence regions")
    return silences


# ═══════════════════════════════════════════════════════════════════════════
#  LEGACY API — Fixed Threshold (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

def detect_silences(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
) -> list[tuple[int, int]]:
    """
    Detect silent portions in an audio file (legacy fixed-threshold API).

    NOTE: For Quranic recitations, prefer detect_silences_adaptive() which
    automatically adapts to the recording's noise characteristics.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)

    Returns:
        List of (start_ms, end_ms) tuples for silent portions
    """
    if use_fast:
        try:
            return _detect_silences_fast(audio_path, min_silence_len, silence_thresh)
        except Exception:
            pass  # Fallback to pydub

    return _detect_silences_pydub(audio_path, min_silence_len, silence_thresh)


def _detect_silences_pydub(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """Pydub-based silence detection (slower but reliable)."""
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_wav(str(audio_path))
    silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return [(s[0], s[1]) for s in silences]


def _detect_silences_fast(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """
    Fast silence detection using librosa + numpy.

    ~10-50x faster than pydub for long files.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None)

    # Convert dB threshold to amplitude ratio
    amplitude_thresh = 10 ** (silence_thresh / 20)

    frame_length = int(sr * 0.01)  # 10ms frames
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize RMS to 0-1 range
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_normalized = rms / rms_max

    is_silent = rms_normalized < amplitude_thresh

    frame_times_ms = (
        librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
    )

    silences: list[tuple[int, int]] = []
    in_silence = False
    silence_start = 0.0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            silence_start = frame_times_ms[i]
        elif not silent and in_silence:
            in_silence = False
            silence_end = frame_times_ms[i]
            duration = silence_end - silence_start
            if duration >= min_silence_len:
                silences.append((int(silence_start), int(silence_end)))

    if in_silence:
        silence_end = frame_times_ms[-1]
        duration = silence_end - silence_start
        if duration >= min_silence_len:
            silences.append((int(silence_start), int(silence_end)))

    return silences


# ═══════════════════════════════════════════════════════════════════════════
#  NON-SILENT CHUNK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_non_silent_chunks(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
) -> list[tuple[int, int]]:
    """
    Detect non-silent (speech) portions in an audio file.

    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)

    Returns:
        List of (start_ms, end_ms) tuples for non-silent portions
    """
    if use_fast:
        try:
            return _detect_non_silent_fast(audio_path, min_silence_len, silence_thresh)
        except Exception:
            pass  # Fallback to pydub

    return _detect_non_silent_pydub(audio_path, min_silence_len, silence_thresh)


def _detect_non_silent_pydub(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """Pydub-based non-silent detection (slower but reliable)."""
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_wav(str(audio_path))
    chunks = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return [(c[0], c[1]) for c in chunks]


def _detect_non_silent_fast(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """
    Fast non-silent chunk detection using librosa + numpy.

    Returns the inverse of silence detection.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None)
    duration_ms = int(len(y) / sr * 1000)

    amplitude_thresh = 10 ** (silence_thresh / 20)

    frame_length = int(sr * 0.01)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
    rms_normalized = rms / rms_max

    is_speech = rms_normalized >= amplitude_thresh

    frame_times_ms = (
        librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000
    )

    chunks: list[tuple[int, int]] = []
    in_speech = False
    speech_start = 0.0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech = True
            speech_start = frame_times_ms[i]
        elif not speech and in_speech:
            in_speech = False
            speech_end = frame_times_ms[i]
            chunks.append((int(speech_start), int(speech_end)))

    if in_speech:
        chunks.append((int(speech_start), duration_ms))

    # Merge chunks that are separated by less than min_silence_len
    if len(chunks) > 1:
        merged = [chunks[0]]
        for start, end in chunks[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_silence_len:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        chunks = merged

    return chunks if chunks else [(0, duration_ms)]


# ═══════════════════════════════════════════════════════════════════════════
#  ENERGY ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def compute_energy_envelope(
    audio_path: str | Path,
    window_ms: int = 50,
) -> list[tuple[float, float]]:
    """
    Compute RMS energy envelope of an audio file.

    Returns a list of (time_seconds, rms_energy) tuples at the given
    window resolution. Useful for detecting local energy minima as
    potential ayah boundary points.

    Args:
        audio_path: Path to audio file
        window_ms: Window size in milliseconds (default 50ms)

    Returns:
        List of (time_sec, rms) tuples
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None)

    frame_length = int(sr * window_ms / 1000)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return [(float(t), float(r)) for t, r in zip(times, rms, strict=False)]


def find_energy_minima(
    envelope: list[tuple[float, float]],
    search_start: float,
    search_end: float,
    top_n: int = 3,
) -> list[float]:
    """
    Find local energy minima within a time range.

    Used to find the best boundary point near an estimated ayah boundary.

    Args:
        envelope: Energy envelope from compute_energy_envelope()
        search_start: Start of search window (seconds)
        search_end: End of search window (seconds)
        top_n: Number of top minima to return

    Returns:
        List of times (seconds) at local energy minima, sorted by energy (lowest first)
    """
    candidates = [(t, e) for t, e in envelope if search_start <= t <= search_end]

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1])
    return [t for t, _ in candidates[:top_n]]


def load_audio_waveform(
    audio_path: str | Path,
    sample_rate: int = 16000,
) -> tuple:
    """
    Load audio waveform for processing.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        Tuple of (waveform_array, sample_rate)
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sample_rate)
    return y, sr


def extract_segment_audio(
    waveform: Any,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
) -> Any:
    """
    Extract a segment from a waveform.

    Args:
        waveform: Audio waveform array
        sample_rate: Sample rate of the waveform
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds

    Returns:
        Segment waveform array
    """
    start_sample = int((start_ms / 1000) * sample_rate)
    end_sample = int((end_ms / 1000) * sample_rate)
    return waveform[start_sample:end_sample]
