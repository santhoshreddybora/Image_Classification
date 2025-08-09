import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import base64
import logging
import mlflow
from flask import request
logging.basicConfig(level=logging.INFO)
logging.info("Your log message")
def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", "outputs")  # fallback to local test
    model_path =model_path = os.path.join(model_dir, "model") # adjust path
    model = mlflow.keras.load_model(model_path)
    logging.info(f"Model loaded from: {model_path}")

def preprocess_data(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    input_arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return input_arr


def run(raw_data):
    try:
        img_bytes=None
        classes = ['brain_glioma', 'brain_menin', 'brain_tumor']

        if hasattr(request,"files") and "file" in request.files:
            image_file= request.files["file"]
            image_bytes=image_file.read()
            logging.info("Image Received from file upload")
        elif raw_data:
            data = json.loads(raw_data)
            image_data = data.get("image_base64")

            if image_data is None:
                return {"error": "Missing 'image_base64' in request."}
            else: 
                # Decode base64 image string
                image_bytes = base64.b64decode(image_data)
                logging.info("Image received from base64 json ")
        if not image_bytes:
            logging.info("No image provided")
            return {"error : No image provided"}

        input_arr=preprocess_data(image_bytes)
        preds = model.predict(input_arr)
        predicted_index = classes[int(np.argmax(preds))]
        confidence = float(np.max(preds))

        return {"class": predicted_index, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
