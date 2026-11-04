import os
import shutil
import asyncio
import subprocess
import json
import re
from datetime import datetime, timezone
from typing import Dict
import numpy as np
import librosa
import torch
from celery import Celery
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_STORAGE = os.path.abspath(os.getenv("STORAGE_PATH", "./storage"))
os.makedirs(BASE_STORAGE, exist_ok=True)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Celery App Setup ---
celery_app = Celery("audio_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# --- DSP & AI Helper Functions ---

def run_demux(input_file: str, output_wav: str):
    """Extracts clean 44.1kHz stereo audio from video or audio containers."""
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_wav
    ]
    run_external_command(cmd, "Could not decode the uploaded audio")

def run_demucs_separation(audio_wav: str, output_dir: str) -> Dict[str, str]:
    """Runs Demucs HT-Demucs 6-stem isolation."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["demucs", "-n", "htdemucs_6s", "--device", DEVICE, "-o", output_dir, audio_wav]
    run_external_command(cmd, "Stem separation failed")
    
    track_name = os.path.splitext(os.path.basename(audio_wav))[0]
    stems_dir = os.path.join(output_dir, "htdemucs_6s", track_name)
    
    stems = {}
    for stem in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
        path = os.path.join(stems_dir, f"{stem}.wav")
        if os.path.exists(path):
            stems[stem] = path
    return stems

def run_external_command(cmd: list[str], failure_message: str) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"Required command '{cmd[0]}' was not found. Install it and add it to PATH.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{failure_message}: {detail}") from error

def analyze_music(audio_wav: str) -> dict:
    """Calculates BPM and musical key via pitch profile correlation."""
    y, sr = librosa.load(audio_wav, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)

    # Key profiling
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    maj_prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_prof = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    best_corr = -1
    detected_key = "C Major"
    for i in range(12):
        c_maj = np.corrcoef(chroma_avg, np.roll(maj_prof, i))[0, 1]
        c_min = np.corrcoef(chroma_avg, np.roll(min_prof, i))[0, 1]
        if c_maj > best_corr:
            best_corr, detected_key = c_maj, f"{pitches[i]} Major"
        if c_min > best_corr:
            best_corr, detected_key = c_min, f"{pitches[i]} Minor"

    return {"bpm": round(bpm, 2), "key": detected_key, "beats": [round(float(t), 2) for t in beats]}

def transcribe_vocals(vocal_path: str) -> list:
    """Performs time-synced lyric extraction using Whisper."""
    from faster_whisper import WhisperModel
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    model = WhisperModel("base", device=DEVICE, compute_type=compute_type)
    audio, _ = librosa.load(vocal_path, sr=16000, mono=True)
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True)
    
    results = []
    for s in segments:
        words = [{"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)} for w in (s.words or [])]
        results.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})
    return results

def save_project_record(work_dir: str, input_path: str, result: dict) -> dict:
    """Persist the processed track and its editable mix settings."""
    project_path = os.path.join(work_dir, "project.json")
    now = datetime.now(timezone.utc).isoformat()
    project = {
        "task_id": os.path.basename(work_dir),
        "title": os.path.basename(input_path),
        "source_filename": os.path.basename(input_path),
        "created_at": now,
        "updated_at": now,
        "metadata": result["metadata"],
        "stems": result["stems"],
        "lyrics": result["lyrics"],
        "volumes": {stem: 1.0 for stem in result["stems"]},
        "muted": {stem: False for stem in result["stems"]},
        "semitones": 0.0,
        "speed": 1.0,
    }
    with open(project_path, "w", encoding="utf-8") as project_file:
        json.dump(project, project_file, ensure_ascii=False)
    return project

# --- Celery Pipeline Task ---

@celery_app.task(bind=True)
def run_audio_pipeline(self, input_path: str, work_dir: str):
    extracted_wav = os.path.join(work_dir, "master.wav")
    stems_dir = os.path.join(work_dir, "stems")

    self.update_state(state="PROGRESS", meta={"stage": "Demuxing Video/Audio...", "progress": 15})
    run_demux(input_path, extracted_wav)

    self.update_state(state="PROGRESS", meta={"stage": "AI Stem Separation (Demucs)...", "progress": 35})
    stems = run_demucs_separation(extracted_wav, stems_dir)
    self.update_state(state="PROGRESS", meta={"stage": "Analyzing Tempo & Key...", "progress": 70})

    music_data = analyze_music(extracted_wav)
    
    lyrics = []
    if "vocals" in stems:
        self.update_state(state="PROGRESS", meta={"stage": "Transcribing Synced Lyrics...", "progress": 85})
        lyrics = transcribe_vocals(stems["vocals"])

    res = {
        "status": "COMPLETED",
        "metadata": music_data,
        "stems": list(stems.keys()),
        "lyrics": lyrics
    }
    save_project_record(work_dir, input_path, res)
    self.update_state(state="SUCCESS", meta=res)
    return res

# --- FastAPI Web Server ---

app = FastAPI(title="Audio Tuner")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ExportPayload(BaseModel):
    task_id: str
    volumes: Dict[str, float]
    semitones: float = 0.0
    speed: float = 1.0

class ProjectSettings(BaseModel):
    volumes: Dict[str, float]
    muted: Dict[str, bool]
    semitones: float = 0.0
    speed: float = 1.0

def get_task_directory(task_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{12}", task_id):
        raise HTTPException(status_code=400, detail="Invalid track ID")
    task_dir = os.path.abspath(os.path.join(BASE_STORAGE, task_id))
    if not os.path.isdir(task_dir):
        raise HTTPException(status_code=404, detail="Track not found")
    return task_dir

def validate_mix_settings(volumes: Dict[str, float], semitones: float, speed: float) -> None:
    if any(stem not in {"vocals", "drums", "bass", "guitar", "piano", "other"} for stem in volumes):
        raise HTTPException(status_code=400, detail="Invalid stem name")
    if any(not 0 <= volume <= 1 for volume in volumes.values()):
        raise HTTPException(status_code=400, detail="Track levels must be between 0 and 1")
    if not -12 <= semitones <= 12:
        raise HTTPException(status_code=400, detail="Pitch must be between -12 and 12 semitones")
    if not 0.5 <= speed <= 2:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5x and 2x")

def read_project(task_dir: str) -> dict:
    project_path = os.path.join(task_dir, "project.json")
    if not os.path.isfile(project_path):
        raise HTTPException(status_code=404, detail="Processed track is not ready")
    try:
        with open(project_path, "r", encoding="utf-8") as project_file:
            project = json.load(project_file)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Could not read the saved track") from error
    if not isinstance(project, dict):
        raise HTTPException(status_code=500, detail="Saved track data is invalid")
    return project

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    task_id = os.urandom(6).hex()
    task_dir = os.path.join(BASE_STORAGE, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    filename = os.path.basename((file.filename or "audio-upload").replace("\\", "/")) or "audio-upload"
    raw_path = os.path.join(task_dir, filename)
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = run_audio_pipeline.apply_async(args=[raw_path, task_dir], task_id=task_id)
    return {"task_id": task.id}

@app.get("/api/library")
async def get_library():
    projects = []
    for task_id in os.listdir(BASE_STORAGE):
        task_dir = os.path.join(BASE_STORAGE, task_id)
        if not re.fullmatch(r"[a-f0-9]{12}", task_id) or not os.path.isdir(task_dir):
            continue
        project_path = os.path.join(task_dir, "project.json")
        if not os.path.isfile(project_path):
            continue
        project = read_project(task_dir)
        projects.append({
            key: project[key]
            for key in ("task_id", "title", "updated_at", "metadata", "stems")
            if key in project
        })
    projects.sort(key=lambda project: project.get("updated_at", ""), reverse=True)
    return projects

@app.get("/api/projects/{task_id}")
async def get_project(task_id: str):
    return read_project(get_task_directory(task_id))

@app.put("/api/projects/{task_id}")
async def update_project(task_id: str, settings: ProjectSettings):
    task_dir = get_task_directory(task_id)
    project = read_project(task_dir)
    validate_mix_settings(settings.volumes, settings.semitones, settings.speed)
    project_stems = set(project.get("stems", []))
    if set(settings.volumes) != project_stems or set(settings.muted) != project_stems:
        raise HTTPException(status_code=400, detail="Mix settings must include every track stem")
    project["volumes"] = settings.volumes
    project["muted"] = settings.muted
    project["semitones"] = settings.semitones
    project["speed"] = settings.speed
    project["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(task_dir, "project.json"), "w", encoding="utf-8") as project_file:
        json.dump(project, project_file, ensure_ascii=False)
    return {"status": "saved"}

@app.websocket("/ws/progress/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            res = AsyncResult(task_id, app=celery_app)
            if res.state == "PROGRESS":
                await websocket.send_json({"state": "PROGRESS", **res.info})
            elif res.state == "SUCCESS":
                await websocket.send_json({"state": "SUCCESS", "result": res.result})
                break
            elif res.state == "FAILURE":
                await websocket.send_json({"state": "FAILURE", "error": str(res.info)})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

@app.get("/api/stems/{task_id}/{stem}")
async def get_stem_audio(task_id: str, stem: str):
    if stem not in {"vocals", "drums", "bass", "guitar", "piano", "other"}:
        raise HTTPException(status_code=404, detail="Stem not found")
    task_dir = get_task_directory(task_id)
    path = os.path.join(task_dir, "stems", "htdemucs_6s", "master", f"{stem}.wav")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Stem not ready")
    return FileResponse(path, media_type="audio/wav")

@app.post("/api/export")
async def export_processed_mix(payload: ExportPayload):
    task_dir = get_task_directory(payload.task_id)
    validate_mix_settings(payload.volumes, payload.semitones, payload.speed)
    stems_dir = os.path.join(task_dir, "stems", "htdemucs_6s", "master")
    
    active_stems = []
    for stem, vol in payload.volumes.items():
        stem_path = os.path.join(stems_dir, f"{stem}.wav")
        if vol > 0.001 and os.path.exists(stem_path):
            active_stems.append((stem_path, vol))

    if not active_stems:
        raise HTTPException(status_code=400, detail="All stems muted")

    inputs, filters = [], []
    for idx, (path, vol) in enumerate(active_stems):
        inputs.extend(["-i", path])
        filters.append(f"[{idx}:a]volume={vol:.2f}[a{idx}]")

    num = len(active_stems)
    mix_chain = "".join([f"[a{i}]" for i in range(num)]) + f"amix=inputs={num}:normalize=0[mixed]"
    
    dsp = []
    if abs(payload.semitones) > 0.01:
        dsp.append(f"rubberband=pitch={2 ** (payload.semitones / 12.0)}")
    if abs(payload.speed - 1.0) > 0.01:
        dsp.append(f"rubberband=tempo={payload.speed}")
        
    dsp_graph = f";[mixed]{','.join(dsp)}[out]" if dsp else ""
    out_label = "[out]" if dsp else "[mixed]"
    final_filter = f"{';'.join(filters)};{mix_chain}{dsp_graph}"

    out_file = f"mix_{payload.task_id}.mp3"
    out_path = os.path.join(task_dir, out_file)
    
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", final_filter, "-map", out_label,
        "-codec:a", "libmp3lame", "-qscale:a", "2", out_path
    ]
    try:
        run_external_command(cmd, "Audio export failed")
    except RuntimeError as error:
        status_code = 503 if "was not found" in str(error) else 500
        raise HTTPException(status_code=status_code, detail=str(error)) from error

    return {"download_url": f"/api/download/{payload.task_id}/{out_file}"}

@app.get("/api/download/{task_id}/{filename}")
async def serve_download(task_id: str, filename: str):
    if os.path.basename(filename) != filename or not filename.endswith(".mp3"):
        raise HTTPException(status_code=404, detail="Export not found")
    task_dir = get_task_directory(task_id)
    path = os.path.join(task_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path, filename=filename, media_type="audio/mpeg")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/styles.css")
async def serve_stylesheet():
    return FileResponse(os.path.join(BASE_DIR, "styles.css"), media_type="text/css")