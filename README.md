# Audio Tuner

Audio Tuner is a local audio workspace for separating a track into stems, tuning and mixing them, and reopening saved projects from the library.

## Run on Windows

1. Install Python 3.10 or newer, FFmpeg (including the `rubberband` filter), and Redis. Redis can be run locally or with Docker:
   ```powershell
   docker run --name audio-tuner-redis -p 6379:6379 redis:7-alpine
   ```
2. From this folder, install the Python packages:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
3. In one terminal, start the Celery worker:
   ```powershell
   celery -A main.celery_app worker --loglevel=INFO --pool=solo
   ```
4. In another terminal, start the web app:
   ```powershell
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
5. Open <http://127.0.0.1:8000>.

The first separation and transcription may take longer while models are downloaded. Projects and mix settings are stored under `storage/`.

If you opened a static preview such as `http://127.0.0.1:8765`, keep the Audio Tuner API running on port `8000`; the preview now sends API and audio requests there. A “server is unreachable” message means the FastAPI app is not running. A processing error after upload usually means Redis or the Celery worker is not running.
