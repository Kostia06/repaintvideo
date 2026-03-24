import os
import uuid
import tempfile

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from style_transfer import StyleTransferEngine, STYLE_MODELS

app = FastAPI(title="RepaintVideo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = StyleTransferEngine()

jobs: dict[str, dict] = {}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "styles": engine.available_styles()}


ALL_STYLE_META: dict[str, dict[str, str]] = {
    "mosaic": {"label": "Mosaic", "description": "Colorful tile mosaic pattern"},
    "candy": {"label": "Candy", "description": "Vibrant candy-colored painting"},
    "rain_princess": {"label": "Rain Princess", "description": "Moody impressionist rain scene"},
    "udnie": {"label": "Udnie", "description": "Abstract cubist fragmentation"},
    "monet": {"label": "Monet", "description": "Soft impressionist brushwork"},
    "starry_night": {"label": "Starry Night", "description": "Swirling Van Gogh texture"},
    "cyberpunk": {"label": "Cyberpunk", "description": "Neon-lit urban grit"},
    "ukiyo_e": {"label": "Ukiyo-e", "description": "Japanese woodblock print"},
    "anime": {"label": "Anime", "description": "Cel-shaded illustration"},
    "watercolor": {"label": "Watercolor", "description": "Soft wet-edge painting"},
    "pixel_art": {"label": "Pixel Art", "description": "Retro game aesthetic"},
    "oil_painting": {"label": "Oil Painting", "description": "Thick impasto brushwork"},
    "pop_art": {"label": "Pop Art", "description": "Bold Warhol-style colors"},
    "sketch": {"label": "Sketch", "description": "Pencil drawing effect"},
    "vintage": {"label": "Vintage", "description": "Retro film nostalgia"},
    "neon_glow": {"label": "Neon Glow", "description": "Glowing rainbow edges"},
}


@app.get("/api/styles")
def get_styles() -> dict:
    available = engine.available_styles()
    return {
        "styles": [
            {"key": k, **meta, "available": k in available}
            for k, meta in ALL_STYLE_META.items()
        ]
    }


@app.post("/api/style/image")
async def style_image(
    file: UploadFile = File(...),
    style: str = Form(...),
) -> Response:
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    try:
        styled = engine.apply_style(frame, style)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _, encoded = cv2.imencode(".jpg", styled, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


def _process_video(job_id: str, input_path: str, style: str, output_path: str) -> None:
    jobs[job_id]["status"] = "processing"

    def on_progress(pct: int) -> None:
        jobs[job_id]["progress"] = pct

    try:
        engine.apply_style_video(input_path, style, output_path, progress_cb=on_progress)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["output_path"] = output_path
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.post("/api/style/video")
async def style_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form(...),
) -> dict:
    job_id = str(uuid.uuid4())
    input_path = os.path.join(tempfile.gettempdir(), f"{job_id}_input.mp4")
    output_path = os.path.join(tempfile.gettempdir(), f"{job_id}_output.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    jobs[job_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(_process_video, job_id, input_path, style, output_path)

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "done": job["status"] == "done",
        "error": job.get("error"),
    }


@app.get("/api/download/{job_id}")
def download(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        return JSONResponse({"error": "Not ready"}, status_code=404)
    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename=f"repaintvideo_{job_id}.mp4",
    )


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "out")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
