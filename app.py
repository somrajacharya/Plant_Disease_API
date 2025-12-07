from fastapi import FastAPI, UploadFile, File
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

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


app = FastAPI()

# Load model

# Load disease details
with open("diseases.json", "r") as f:
    DISEASE_DATA = json.load(f)
    DISEASE_DATA = {int(k): v for k, v in DISEASE_DATA.items()}

IMG_SIZE = (224, 224)  # change if your model uses different size

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

@app.get("/")
def home():
    return {"message": "Plant Disease API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
