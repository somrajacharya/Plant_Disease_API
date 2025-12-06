from fastapi import FastAPI, UploadFile, File # (KEEP)
# import uvicorn # Not needed if running via CMD
# import tensorflow as tf # <-- COMMENT OUT
# import numpy as np # <-- COMMENT OUT
from PIL import Image # (KEEP)
import io # (KEEP)
# import json # <-- COMMENT OUT
import os # (KEEP)
# import gdown # <-- COMMENT OUT

# --- OPTIMIZATION: Suppress CUDA/GPU errors on Railway's CPU host ---
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------------------------------

# MODEL_PATH = "plant_disease_prediction_model.h5" # <-- COMMENT OUT
# MODEL_URL = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf" # <-- COMMENT OUT

# # Download model if it doesn't exist 
# if not os.path.exists(MODEL_PATH): # <-- COMMENT OUT
#     print("Downloading model from Google Drive...") # <-- COMMENT OUT
#     gdown.download(MODEL_URL, MODEL_PATH, quiet=False) # <-- COMMENT OUT

# # Load the model (This runs ONCE when the container starts)
# # model = tf.keras.models.load_model(MODEL_PATH, compile=False) # <-- COMMENT OUT
# print("✨ Model loaded successfully. Starting API.") # <-- COMMENT OUT

app = FastAPI() # (KEEP)

# Load disease details (Runs once at startup)
# with open("diseases.json", "r") as f:
#     DISEASE_DATA = json.load(f)
#     DISEASE_DATA = {int(k): v for k, v in DISEASE_DATA.items()}
DISEASE_DATA = { # (KEEP)
    0: {"name": "Test Healthy", "description": "This is a temporary placeholder to test the API server startup.", "cause": "", "solution": "", "prevention": ""} # (KEEP)
} # (KEEP)

IMG_SIZE = (224, 224) # (KEEP)

def preprocess(img_bytes): # (KEEP)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB") # (KEEP)
    img = img.resize(IMG_SIZE) # (KEEP)
    # img = np.array(img) / 255.0 # <-- COMMENT OUT (since numpy is gone)
    # img = np.expand_dims(img, axis=0) # <-- COMMENT OUT
    return "TEST_SUCCESS" # Return a simple string instead

@app.post("/predict")
async def predict(file: UploadFile = File(...)): # (KEEP)
    # img_bytes = await file.read() # <-- COMMENT OUT
    # img = preprocess(img_bytes) # <-- COMMENT OUT

    # preds = model.predict(img)[0] # <-- COMMENT OUT
    # class_index = int(np.argmax(preds)) # <-- COMMENT OUT
    # confidence = float(np.max(preds)) # <-- COMMENT OUT

    # info = DISEASE_DATA[class_index] # <-- COMMENT OUT
    
    # Return dummy data to ensure the server starts
    info = DISEASE_DATA[0] 

    return { # (KEEP)
        "class_index": 0, # (KEEP)
        "disease_name": info["name"], # (KEEP)
        "description": info["description"], # (KEEP)
        "cause": info["cause"], # (KEEP)
        "solution": info["solution"], # (KEEP)
        "prevention": info["prevention"], # (KEEP)
        "confidence": 0.999 # (KEEP)
    } # (KEEP)

@app.get("/")
def home(): # (KEEP)
    return {"message": "Plant Disease API is running!"} # (KEEP)
