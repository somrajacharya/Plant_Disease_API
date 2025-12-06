#!/bin/bash

# Define constants (must match app.py)
MODEL_PATH="plant_disease_prediction_model.h5"
MODEL_URL="https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"

# 1. Download the model if it's not already present.
if [ ! -f "$MODEL_PATH" ]; then
    echo "--- Model not found. Starting robust download ---"
    # Use Python/gdown command to perform the download
    python -c "import gdown; gdown.download('$MODEL_URL', '$MODEL_PATH', quiet=False)"
    echo "--- Download complete. Starting API. ---"
else
    echo "--- Model found on disk. Skipping download. ---"
fi

# 2. Start the Uvicorn server using your app:app reference.
# The server will now start quickly because the heavy file is already on disk.
uvicorn app:app --host 0.0.0.0 --port 8000