FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --frozen-lockfile
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

COPY --from=frontend-builder /app/frontend/out ./frontend/out

COPY scripts/download_models.py ./scripts/download_models.py

RUN mkdir -p models/weights

WORKDIR /app/backend

EXPOSE 7860

CMD python ../scripts/download_models.py && uvicorn main:app --host 0.0.0.0 --port 7860
