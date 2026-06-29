import os
import uuid
import gc
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import soundfile as sf
import torch

# WaqfBackend imports
from waqf_backend.transcription.whisperx import Whisperx
from waqf_backend.core.aligner import align
from waqf_backend.models.ayah import Ayah
from waqf_backend.formatters import format_alignment_results

from waqf_backend.data import load_surah_ayahs
from waqf_backend.transcription.silence import detect_silences_adaptive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WaqfBackend Alignment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_audio", exist_ok=True)

jobs: dict = {}
_executor = ThreadPoolExecutor(max_workers=1)

def _run_job(job_id: str, file_path: str, surah_number: int, silence_sensitivity: float, silence_engine: str = "librosa"):
    """Runs the WaqfBackend alignment pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "processing"
        
        # 1. Load reference ayahs internally
        ayahs = load_surah_ayahs(surah_number)
        if not ayahs:
            raise ValueError(f"Could not load reference ayahs for surah {surah_number}.")
        
        logger.info(f"[Job {job_id[:8]}] Surah: {surah_number}, Ayahs: {len(ayahs)}")

        # 2. Transcribe
        with Whisperx(model_name="large-v2", device="cuda") as transcriber:
            segments = transcriber.transcribe(file_path, surah_id=surah_number, silence_percentile=silence_sensitivity, silence_engine=silence_engine)
            
        logger.info(f"[Job {job_id[:8]}] Found {len(segments)} segments.")
        
        # Detect raw silences for robust frontend Waqf segmentation
        from waqf_backend.transcription.silence import detect_silences_adaptive, detect_silences_vad
        if silence_engine == "silero":
            raw_silences_ms = detect_silences_vad(
                file_path,
                min_silence_len=150,
            )
        else:
            raw_silences_ms = detect_silences_adaptive(
                file_path,
                min_silence_len=150,
                percentile=silence_sensitivity,
                smooth_kernel=7,
                merge_gap_ms=80,
            )
        silences_sec = [[s[0]/1000.0, s[1]/1000.0] for s in raw_silences_ms]

        # 3. Align using WaqfBackend's core
        results = align(
            audio_path=file_path,
            segments=segments,
            ayahs=ayahs,
            strategy="auto"
        )
        
        logger.info(f"[Job {job_id[:8]}] Aligned {len(results)} ayahs.")

        # 4. Format results to match the API expectation
        response_data = []
        
        # Build a lookup for original segment words
        seg_words_by_id = {s.id: getattr(s, "words", None) for s in segments if hasattr(s, 'id')}
        
        for r in results:
            ayah_data = {
                "ayah_number": r.ayah.ayah_number,
                "start_time": r.start_time,
                "end_time": r.end_time
            }
            
            words = getattr(r, "words", None)
            if not words:
                words = seg_words_by_id.get(r.ayah.ayah_number)
                
            if words:
                ayah_data["words"] = [{"word": getattr(w, "word", w), "start": w.start, "end": w.end, "original_start": getattr(w, "original_start", None), "original_end": getattr(w, "original_end", None)} for w in words]
            response_data.append(ayah_data)

        jobs[job_id] = {
            "status": "success",
            "data": response_data,
            "silences": silences_sec,
            "error": None,
        }
        logger.info(f"[Job {job_id[:8]}] ✓ done")

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            "status": "error",
            "data": None,
            "error": str(e),
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@app.post("/align/{surah_number}")
async def align_audio(
    surah_number: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    riwaya: str = Form("hafs"),
    silence_sensitivity: float = Form(15.0),
    silence_engine: str = Form("librosa")
):
    job_id = str(uuid.uuid4())
    file_path = f"temp_audio/temp_{job_id}_{surah_number}.mp3"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued", "data": None, "error": None}

    background_tasks.add_task(
        lambda: _executor.submit(_run_job, job_id, file_path, surah_number, silence_sensitivity, silence_engine)
    )

    return JSONResponse({
        "status": "queued",
        "job_id": job_id,
        "message": "بدأت المهمة وسيتم فحصها تلقائياً.",
    })

@app.get("/align/status/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "المهمة غير موجودة"}, status_code=404)

    if job["status"] == "success":
        return JSONResponse({
            "status": "success",
            "data": job["data"],
            "silences": job.get("silences", [])
        })
    elif job["status"] == "error":
        return JSONResponse({"status": "error", "message": job["error"]}, status_code=500)
    else:
        return JSONResponse({
            "status": job["status"],
            "message": "المعالجة مستمرة، يرجى الانتظار..."
        })

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
