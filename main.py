from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

# Load model YOLO sekali di awal
try:
    model = YOLO("best.pt")  # pastikan file ini ada di folder yang sama
except Exception as e:
    raise RuntimeError(f"Gagal load model YOLO: {e}")

# class id sesuai training kamu
CLASS_RUSAK_ID = 1  # 0 = bagus, 1 = rusak

class PredictionRequest(BaseModel):
    image_url: HttpUrl

class PredictionResponse(BaseModel):
    status: str           # "Baik" atau "Rusak"
    has_rusak: bool
    confidence: float | None = None

def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal download gambar: {e}")

@app.get("/")
def root():
    return {"message": "SMART Bushing API is running"}

@app.post("/cek_bushing", response_model=PredictionResponse)
def cek_bushing(req: PredictionRequest):
    # 1. Ambil gambar dari URL
    img = load_image_from_url(req.image_url)

    # 2. Inference YOLO
    results = model(img)

    has_rusak = False
    max_conf = 0.0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == CLASS_RUSAK_ID:
                has_rusak = True
                if conf > max_conf:
                    max_conf = conf

    status = "Rusak" if has_rusak else "Baik"

    return PredictionResponse(
        status=status,
        has_rusak=has_rusak,
        confidence=max_conf if has_rusak else None
    )
