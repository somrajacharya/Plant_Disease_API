from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import gdown

MODEL_PATH = "plant_disease_prediction_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
IMG_SIZE = (224, 224)

# Load disease details (no change needed here)
with open("diseases.json", "r") as f:
    DISEASE_DATA = json.load(f)
    DISEASE_DATA = {int(k): v for k, v in DISEASE_DATA.items()}

# Model variable is declared but not assigned here
model = None

# Download model if it doesn't exist (no change needed here)
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# 1. CRITICAL OPTIMIZATION: Use lifespan to load model outside global scope
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Model loading happens when the app starts up
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    yield
    # Clean up on shutdown (optional but good practice)
    model = None

# Pass the lifespan function to FastAPI
app = FastAPI(lifespan=lifespan)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess(img_bytes)

    # Use the globally available model loaded via lifespan
    preds = model.predict(img)[0]
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    info = DISEASE_DATA[class_index]

    return {
        "class_index": class_index,
        "disease_name": info["name"],
        "description": info["description"],
        "cause": info["cause"],
        "solution": info["solution"],
        "prevention": info["prevention"],
        "confidence": confidence
    }

# 2. MINOR ENHANCEMENT: Improve home endpoint
@app.get("/")
def home():
    disease_list = [{"id": k, "name": v["name"]} for k, v in DISEASE_DATA.items()]
    return {
        "message": "Plant Disease API is running!",
        "model_loaded_status": "Successfully loaded via lifespan" if model is not None else "Loading...",
        "supported_disease_count": len(disease_list)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
