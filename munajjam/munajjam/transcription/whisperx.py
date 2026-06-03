import gc
import re
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
try:
    import omegaconf
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
except Exception:
    pass
import whisperx
from ctc_segmentation import CtcSegmentationParameters, ctc_segmentation
from rapidfuzz import fuzz
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path

from munajjam.core.arabic import detect_segment_type
from munajjam.models import Segment, WordTimestamp, SegmentType
from munajjam.transcription.base import BaseTranscriber
from munajjam.data import load_surah_ayahs
from munajjam.config import get_settings


class Whisperx(BaseTranscriber):
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        
        settings = get_settings()
        self.wav2vec2_model_id = settings.wav2vec2_model_id

        self.whisper_model = None
        self.wav2vec2_model = None
        self.wav2vec2_processor = None

    def _normalize_arabic(self, text: str, for_ctc: bool = False) -> str:
        text = re.sub(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
        if for_ctc:
            text = re.sub(r'[أإآٱ]', 'ا', text)
            text = re.sub(r'[^\u0621-\u064A\s]', '', text)
        return text.strip()

    def _load_wav2vec2(self):
        if not self.wav2vec2_model:
            print(f"Loading Wav2Vec2 model from {self.wav2vec2_model_id}...")
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_id)
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_model_id).to(self.device)

    def align_ctc(self, audio_data: np.ndarray, sr: int, words: list[str], offset: float = 0.0) -> list[dict]:
        self._load_wav2vec2()
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        duration = len(audio_data) / sr
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        chunk_size = 30 * 16000
        all_log_probs = []

        with torch.inference_mode():
            for i in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[i : i + chunk_size]
                if len(chunk) < 400: continue
                logits = self.wav2vec2_model(chunk.unsqueeze(0)).logits
                log_probs = torch.log_softmax(logits, dim=-1).cpu()
                all_log_probs.append(log_probs)
            
            if not all_log_probs:
                return []
            combined_log_probs = torch.cat(all_log_probs, dim=1)[0].numpy()

        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = duration / combined_log_probs.shape[0]
        
        clean_words = [self._normalize_arabic(w, for_ctc=True) for w in words]
        clean_words = [w for w in clean_words if w]
        
        if not clean_words:
            return []

        try:
            results = ctc_segmentation(config, combined_log_probs, clean_words)
        except Exception as e:
            print(f"[CTC-Seg] Local Error: {e}")
            return []

        word_alignments = []
        for i, segment in enumerate(results):
            word_alignments.append({
                "word":       words[i], 
                "start":      round(float(segment[0]) + offset, 3),
                "end":        round(float(segment[1]) + offset, 3),
                "confidence": round(min(1.0, float(np.exp(segment[2]))), 2),
            })
        return word_alignments

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []
            
        ref_words = []
        for ayah in ayahs:
            for w in ayah.text.split():
                ref_words.append(w)
        
        if not self.whisper_model:
            print(f"Loading WhisperX model {self.model_name}...")
            self.whisper_model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type, language="ar")
        
        audio = whisperx.load_audio(str(audio_path))
        result = self.whisper_model.transcribe(audio, batch_size=batch_size)
        
        model_a, metadata = whisperx.load_align_model(language_code="ar", device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        extracted_words = []
        for segment in result["segments"]:
            if "words" in segment:
                for w in segment["words"]:
                    if "start" in w and "end" in w:
                        extracted_words.append({
                            "word": w["word"],
                            "start": w["start"],
                            "end": w["end"],
                            "confidence": w.get("score", 0.9)
                        })

        n = len(ref_words)
        m = len(extracted_words)
        dp = np.zeros((n + 1, m + 1))
        
        for i in range(1, n + 1):
            rw = self._normalize_arabic(ref_words[i-1], for_ctc=True)
            for j in range(1, m + 1):
                ew = self._normalize_arabic(extracted_words[j-1]["word"], for_ctc=True)
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.6: match_score = -1.0
                dp[i][j] = max(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1] + match_score
                )
        
        mapped_alignments = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = self._normalize_arabic(ref_words[i-1], for_ctc=True)
            ew = self._normalize_arabic(extracted_words[j-1]["word"], for_ctc=True)
            match_score = fuzz.ratio(rw, ew) / 100.0
            
            if match_score >= 0.6 and dp[i][j] == dp[i-1][j-1] + match_score:
                mapped_alignments[i-1] = extracted_words[j-1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j]:
                i -= 1
            else:
                j -= 1
        
        w_alignments = []
        for k in range(n):
            if mapped_alignments[k]:
                w_alignments.append({
                    "word": ref_words[k],
                    "start": mapped_alignments[k]["start"],
                    "end": mapped_alignments[k]["end"],
                    "confidence": mapped_alignments[k]["confidence"]
                })
            else:
                prev_end = w_alignments[-1]["end"] if w_alignments else 0
                w_alignments.append({
                    "word": ref_words[k],
                    "start": prev_end,
                    "end": prev_end + 0.1,
                    "confidence": 0.0
                })

        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        speech, sr = sf.read(str(audio_path), dtype='float32')
        if len(speech.shape) > 1: speech = speech.mean(axis=1)
        c_alignments = self.align_ctc(speech, sr, ref_words)
        
        final_alignments = []
        c_idx = 0
        
        for w_entry in w_alignments:
            word_text = self._normalize_arabic(w_entry["word"], for_ctc=True)
            best_ctc = None
            min_dist = 0.6 
            
            search_start = max(0, c_idx - 5)
            search_end = min(len(c_alignments), c_idx + 10)
            
            for k in range(search_start, search_end):
                c_entry = c_alignments[k]
                c_text = self._normalize_arabic(c_entry["word"], for_ctc=True)
                
                if word_text == c_text:
                    dist = abs(((w_entry["start"] + w_entry["end"])/2) - ((c_entry["start"] + c_entry["end"])/2))
                    if dist < min_dist:
                        best_ctc = c_entry
                        min_dist = dist
                        c_idx = k + 1 
                        break
            
            if best_ctc and best_ctc["confidence"] > 0.4:
                refined_start = best_ctc["start"]
                refined_end = best_ctc["end"]
                
                if abs(refined_start - w_entry["start"]) > 0.5:
                    refined_start = w_entry["start"]
                if abs(refined_end - w_entry["end"]) > 0.5:
                    refined_end = w_entry["end"]

                final_alignments.append({
                    "word": w_entry["word"],
                    "start": round(refined_start, 3),
                    "end": round(refined_end, 3),
                    "confidence": max(w_entry["confidence"], best_ctc["confidence"])
                })
            else:
                final_alignments.append(w_entry)

        try:
            total_duration = sf.info(str(audio_path)).duration
        except Exception:
            total_duration = final_alignments[-1]["end"] + 2.0

        for k in range(len(final_alignments)):
            if k > 0:
                if final_alignments[k]["start"] < final_alignments[k-1]["end"]:
                    final_alignments[k]["start"] = final_alignments[k-1]["end"]
            
            if k < len(final_alignments) - 1:
                next_start = final_alignments[k+1]["start"]
                current_end = final_alignments[k]["end"]
                gap = next_start - current_end
                if gap > 0.1:
                    final_alignments[k]["end"] = round(next_start - 0.1, 3)
            else:
                final_alignments[k]["end"] = round(total_duration, 3)

            if final_alignments[k]["end"] <= final_alignments[k]["start"]:
                final_alignments[k]["end"] = round(final_alignments[k]["start"] + 0.1, 3)

        word_idx = 0
        segments = []
        
        for ayah in ayahs:
            ayah_words_count = len(ayah.text.split())
            ayah_alignments = final_alignments[word_idx : word_idx + ayah_words_count]
            word_idx += ayah_words_count
            
            if not ayah_alignments:
                continue
                
            words = []
            avg_conf = 0.0
            for wa in ayah_alignments:
                words.append(
                    WordTimestamp(
                        word=wa["word"],
                        start=wa["start"],
                        end=wa["end"],
                        probability=wa["confidence"]
                    )
                )
                avg_conf += wa["confidence"]
            
            if words:
                avg_conf /= len(words)
                segments.append(
                    Segment(
                        id=ayah.ayah_number,
                        surah_id=surah_id,
                        start=words[0].start,
                        end=words[-1].end,
                        text=ayah.text,
                        type=SegmentType.AYAH,
                        words=words,
                        confidence=avg_conf
                    )
                )
            
        return segments
