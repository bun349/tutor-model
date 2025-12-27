"""
FastAPI server gabungan:
- HTTP API: Dual Classifier (Question Type + Empathy)
- WebSocket: YOLO detection dari stream kamera

Run:
    uvicorn predict_api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO
from PIL import Image
import time
import os
import io, json, base64
from collections import defaultdict
import uuid

# ======================================================================
# APP & CORS
# ======================================================================

app = FastAPI(
    title="Counseling & Vision API",
    description="HTTP Dual Classifier + WebSocket YOLO",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # sesuaikan kalau mau lebih ketat
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# CONFIG & GLOBALS
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NLP model paths
QUESTION_MODEL_PATH = os.path.join(BASE_DIR, "indobert_counseling_v2")
EMPATHY_MODEL_PATH = os.path.join(BASE_DIR, "empathy_model2_finetuned")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "face_emotion_detection", "best.pt")
MAX_LENGTH = 128

# Label mappings
QUESTION_LABELS = {0: 'Terbuka', 1: 'Sugestif', 2: 'Tertutup', 3: 'Reflektif'}
EMPATHY_LABELS = {0: 'Empatik', 1: 'Netral', 2: 'Judgemental'}

# Global NLP vars
question_tokenizer = None
question_model = None
empathy_tokenizer = None
empathy_model = None
device = None

# Global YOLO model
yolo_model = None

# ======================================================================
# REQUEST / RESPONSE MODELS (NLP)
# ======================================================================

class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1)

class DualPredictRequest(BaseModel):
    text: str = Field(..., description="Text to classify with both models")

class DualPredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts for dual classification")

class QuestionResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

class EmpathyResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]

class DualPredictionResult(BaseModel):
    text: str
    question_type: QuestionResult
    empathy_level: EmpathyResult
    processing_time_ms: float

class DualPredictResponse(BaseModel):
    success: bool
    data: DualPredictionResult

class DualPredictBatchResponse(BaseModel):
    success: bool
    data: List[DualPredictionResult]
    total_items: int
    total_processing_time_ms: float

# ======================================================================
# STARTUP: LOAD SEMUA MODEL
# ======================================================================

@app.on_event("startup")
async def load_models():
    global question_tokenizer, question_model
    global empathy_tokenizer, empathy_model
    global device, yolo_model

    print("="*70)
    print("Loading NLP & YOLO models...")
    print("="*70)

    # Device NLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device NLP: {device}")

    # 1. Question Type Model
    try:
        print("\n1. Loading Question Type Model...")
        question_tokenizer = AutoTokenizer.from_pretrained(QUESTION_MODEL_PATH)
        question_model = AutoModelForSequenceClassification.from_pretrained(QUESTION_MODEL_PATH)
        question_model.to(device)
        question_model.eval()
        print("   ✓ Question Type Model loaded")
    except Exception as e:
        print("   ✗ Failed to load Question Type Model:", e)
        question_model = None

    # 2. Empathy Model
    try:
        print("\n2. Loading Empathy Model...")
        empathy_tokenizer = AutoTokenizer.from_pretrained(EMPATHY_MODEL_PATH)
        empathy_model = AutoModelForSequenceClassification.from_pretrained(EMPATHY_MODEL_PATH)
        empathy_model.to(device)
        empathy_model.eval()
        print("   ✓ Empathy Model loaded")
    except Exception as e:
        print("   ✗ Failed to load Empathy Model:", e)
        empathy_model = None

    # 3. YOLO Model
    try:
        print("\n3. Loading YOLO Model...")
        # sesuaikan path model YOLO kamu
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("   ✓ YOLO Model loaded")
    except Exception as e:
        print("   ✗ Failed to load YOLO model:", e)
        yolo_model = None

    print("\n" + "="*70)
    print("All models attempted to load.")
    print("="*70 + "\n")

# ======================================================================
# NLP PREDICTION FUNCTIONS
# ======================================================================

def predict_question_type(text: str) -> QuestionResult:
    if question_model is None:
        raise HTTPException(status_code=503, detail="Question model not loaded")

    inputs = question_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = question_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    all_probs = {
        QUESTION_LABELS[i]: float(probs[0][i])
        for i in range(len(QUESTION_LABELS))
    }

    return QuestionResult(
        label=QUESTION_LABELS[predicted_class],
        label_id=predicted_class,
        confidence=confidence,
        probabilities=all_probs
    )

def predict_empathy(text: str) -> EmpathyResult:
    if empathy_model is None:
        raise HTTPException(status_code=503, detail="Empathy model not loaded")

    inputs = empathy_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = empathy_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    all_probs = {
        EMPATHY_LABELS[i]: float(probs[0][i])
        for i in range(len(EMPATHY_LABELS))
    }

    return EmpathyResult(
        label=EMPATHY_LABELS[predicted_class],
        label_id=predicted_class,
        confidence=confidence,
        probabilities=all_probs
    )

# ======================================================================
# ROOT & HEALTH (GABUNG)
# ======================================================================

@app.get("/")
async def root():
    return {
        "service": "Counseling & Vision API",
        "version": "1.0",
        "status": "running",
        "models": {
            "question_type": {
                "loaded": question_model is not None,
                "labels": list(QUESTION_LABELS.values())
            },
            "empathy": {
                "loaded": empathy_model is not None,
                "labels": list(EMPATHY_LABELS.values())
            },
            "yolo": {
                "loaded": yolo_model is not None
            }
        },
        "endpoints": {
            "dual_predict": "/predict-dual",
            "dual_predict_batch": "/predict-dual-batch",
            "question_only": "/predict-question",
            "empathy_only": "/predict-empathy",
            "ws_yolo": "/ws",
            "health": "/health",
            "routes": "/routes",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "nlp_models": {
            "question_type": question_model is not None,
            "empathy": empathy_model is not None
        },
        "vision_models": {
            "yolo": yolo_model is not None
        },
        "device_nlp": str(device) if device else "not initialized"
    }

# ======================================================================
# NLP ENDPOINTS (HTTP)
# ======================================================================

@app.post("/predict-dual", response_model=DualPredictResponse)
async def predict_dual(request: DualPredictRequest):
    start_time = time.time()
    try:
        q_res = predict_question_type(request.text)
        e_res = predict_empathy(request.text)
        processing_time = (time.time() - start_time) * 1000

        result = DualPredictionResult(
            text=request.text,
            question_type=q_res,
            empathy_level=e_res,
            processing_time_ms=processing_time
        )

        return DualPredictResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-dual-batch", response_model=DualPredictBatchResponse)
async def predict_dual_batch(request: DualPredictBatchRequest):
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")

    start_time = time.time()
    results = []

    try:
        for text in request.texts:
            t_start = time.time()
            q_res = predict_question_type(text)
            e_res = predict_empathy(text)
            t_time = (time.time() - t_start) * 1000

            results.append(
                DualPredictionResult(
                    text=text,
                    question_type=q_res,
                    empathy_level=e_res,
                    processing_time_ms=t_time
                )
            )

        total_time = (time.time() - start_time) * 1000
        return DualPredictBatchResponse(
            success=True,
            data=results,
            total_items=len(results),
            total_processing_time_ms=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-question")
async def predict_question_only(request: PredictRequest):
    start_time = time.time()
    try:
        res = predict_question_type(request.text)
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": True,
            "data": {
                "text": request.text,
                **res.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-empathy")
async def predict_empathy_only(request: PredictRequest):
    start_time = time.time()
    try:
        res = predict_empathy(request.text)
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": True,
            "data": {
                "text": request.text,
                **res.dict(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================================
# WEBSOCKET YOLO
# ======================================================================

@app.websocket("/ws")
async def ws_yolo(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to YOLO WebSocket")

    if yolo_model is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "YOLO model not loaded"
        }))
        await websocket.close()
        return

    session_id = str(uuid.uuid4())
    label_counts = defaultdict(int)
    total_frames = 0

    try:
        while True:
            text_data = await websocket.receive_text()
            msg = json.loads(text_data)

            if msg["type"] == "frame":
                data_url = msg["image"]
                header, encoded = data_url.split(",", 1)
                img_bytes = base64.b64decode(encoded)

                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                results = yolo_model.predict(img, conf=0.5, imgsz=640)
                r = results[0]

                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    label_counts[label] += 1

                total_frames += 1

            elif msg["type"] == "finish":
                summary = {
                    "session_id": session_id,
                    "total_frames": total_frames,
                    "labels": [
                        {"label": lbl, "count": cnt}
                        for lbl, cnt in label_counts.items()
                    ]
                }

                await websocket.send_text(json.dumps({
                    "type": "summary",
                    "data": summary
                }))

                await websocket.close()
                print("Session done. WebSocket closed.")
                break

    except WebSocketDisconnect:
        print("Client disconnected unexpectedly")
    except Exception as e:
        print("Error in WebSocket:", e)

# ======================================================================
# ROUTE LIST
# ======================================================================

@app.get("/routes")
def list_routes():
    return [{"path": r.path, "name": r.name} for r in app.router.routes]

# ======================================================================
# MAIN (optional kalau run langsung)
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
