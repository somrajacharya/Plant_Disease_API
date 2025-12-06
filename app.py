from fastapi import FastAPI, UploadFile, File
import uvicorn

import tensorflow_cpu as tf
import numpy as np
from PIL import Image
import io
import json
import os
import gdown

# --- OPTIMIZATION: Suppress CUDA/GPU errors on Railway's CPU host ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------------------------------

MODEL_PATH = "plant_disease_prediction_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"

# --- DEFERRED LOADING SETUP ---
# 1. model is initialized as None
model = None 

# 2. Function to handle download and load
# 2. Function to handle load only (Download is handled by start.sh)
def load_model_once():
    global model
    if model is not None:
        return model
        
    print("--- DEFERRED MODEL LOAD INITIATED (FROM DISK) ---")

    # Load the model inside the function
    try:
        # The .h5 file is guaranteed to be on disk by the start.sh script
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✨ Model loaded successfully (Deferred).")
        return model
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model from disk: {e}")
        # If it fails to load from disk, we must crash
        raise

app = FastAPI()

# Using DUMMY DATA for now (we can fix diseases.json once the server runs)
# Load disease details (Runs once at startup)
with open("diseases.json", "r") as f:
    DISEASE_DATA = json.load(f)
    DISEASE_DATA = {int(k): v for k, v in DISEASE_DATA.items()}

IMG_SIZE = (224, 224) 

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3. Call the function to ensure the model is loaded here!
    loaded_model = load_model_once() 

    img_bytes = await file.read()
    img = preprocess(img_bytes)

    preds = loaded_model.predict(img)[0]
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    info = DISEASE_DATA.get(class_index, DISEASE_DATA[0]) 

    return {
        "class_index": class_index,
        "disease_name": info["name"],
        "description": info["description"],
        "cause": info["cause"],
        "solution": info["solution"],
        "prevention": info["prevention"],
        "confidence": confidence
    }

@app.get("/")
def home():
    return {"message": "Plant Disease API is running! (Model loads on first /predict call)"}


