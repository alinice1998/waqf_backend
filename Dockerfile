FROM python:3.11-slim

# Install system dependencies (ffmpeg is required for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install Munajjam and WhisperX
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git && \
    pip install --no-cache-dir . && \
    pip install "numpy<2"

# Install API Server dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Required directories
RUN mkdir -p /app/temp_audio

EXPOSE 8000

# Start the API server
ENTRYPOINT ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
