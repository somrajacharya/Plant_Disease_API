from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import gdown

# --- OPTIMIZATION: Suppress CUDA/GPU errors on Railway's CPU host ---
# These lines help prevent noisy/error logs on non-GPU environments
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------------------------------

MODEL_PATH = "plant_disease_prediction_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"

# Download model if it doesn't exist (Only runs once during startup/boot)
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        # Re-raise the exception to crash fast, so you see the error
        raise

# Load the model (This runs ONCE when the container starts)
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✨ Model loaded successfully. Starting API.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load model: {e}")
    # Re-raise the exception to crash fast
    raise


app = FastAPI()

# Load disease details (Using DUMMY DATA for now, to focus only on model crash)
# The old code is commented out below:
# with open("diseases.json", "r") as f:
#     DISEASE_DATA = json.load(f)
#     DISEASE_DATA = {int(k): v for k, v in DISEASE_DATA.items()}
DISEASE_DATA = {
    0: {"name": "Test Healthy", "description": "Temp description", "cause": "Temp cause", "solution": "Temp solution", "prevention": "Temp prevention"},
    # Add a few more index placeholders if your model output is large
    1: {"name": "Placeholder 1", "description": "...", "cause": "...", "solution": "...", "prevention": "..."},
    2: {"name": "Placeholder 2", "description": "...", "cause": "...", "solution": "...", "prevention": "..."}
}

IMG_SIZE = (224, 224) 

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

    preds = model.predict(img)[0]
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Safely get index, using 0 if model output index is not in dummy data
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
    return {"message": "Plant Disease API is running!"}
