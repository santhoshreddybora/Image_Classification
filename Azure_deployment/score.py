import os
import io
import json
import logging
import base64
import datetime
from azure.storage.blob import BlobServiceClient
from flask import request
import numpy as np
import tensorflow as tf
from PIL import Image
import mlflow
from opencensus.ext.azure.log_exporter import AzureLogHandler

# ✅ Global init
model = None
blob_container = None
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init():
    global model, blob_container

    # --- Load model ---
    model_dir = os.getenv("AZUREML_MODEL_DIR", "outputs")
    model_path = os.path.join(model_dir, "model")
    model = mlflow.keras.load_model(model_path)
    logger.info(f" Model loaded from: {model_path}")

    # --- Setup Application Insights logging ---
    if "APPINSIGHTS_INSTRUMENTATIONKEY" in os.environ:
        ikey = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")
        logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={ikey}'))
        logger.info(" Application Insights connected")

    # --- Setup Azure Blob client ---
    try:
        blob_conn_str = os.getenv("AZURE_BLOB_CONN_STR")
        container_name = "prediction-logs"
        blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
        container_client = blob_service_client.get_container_client(container_name)

        if not container_client.exists():
            container_client.create_container()

        blob_container = container_client
        logger.info(f" Connected to Azure Blob container: {container_name}")
    except Exception as e:
        logger.error(f" Failed to connect to Azure Blob Storage: {e}")
        blob_container = None


def preprocess_data(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    return arr


def run(raw_data):
    try:
        classes = ["brain_glioma", "brain_menin", "brain_tumor"]
        img_bytes = None

        # --- Get image data ---
        if hasattr(request, "files") and "file" in request.files:
            image_file = request.files["file"]
            image_bytes = image_file.read()
            logger.info("Image received via file upload")
        elif raw_data:
            data = json.loads(raw_data)
            image_data = data.get("image_base64")
            if not image_data:
                raise ValueError("Missing 'image_base64'")
            image_bytes = base64.b64decode(image_data)
            logger.info("Image received via base64 JSON")
        else:
            raise ValueError("No image provided")

        # --- Predict ---
        input_arr = preprocess_data(image_bytes)
        preds = model.predict(input_arr)
        predicted_class = classes[int(np.argmax(preds))]
        confidence = float(np.max(preds))
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Prepare log entry ---
        log_entry = {
            "timestamp": timestamp,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "workspace": os.getenv("AZUREML_WORKSPACE_NAME", "local-test"),
        }

        # --- Save prediction results & image ---
        if blob_container:
            # Folder for JSON and image
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

            # 1️⃣ Append to daily JSON log
            blob_name = f"prediction_logs/{today}.json"
            existing_data = []

            try:
                blob_data = blob_container.download_blob(blob_name).readall()
                existing_data = json.loads(blob_data.decode("utf-8"))
            except Exception:
                pass  # first entry of the day

            existing_data.append(log_entry)
            blob_container.upload_blob(
                blob_name,
                data=json.dumps(existing_data, indent=2),
                overwrite=True,
            )
            logger.info(f" Logged prediction to {blob_name}")

            # 2️⃣ Save the image itself
            image_folder = f"prediction_logs/images/{today}/"
            image_name = f"img_{timestamp.replace(':','-')}_{predicted_class}.jpg"
            blob_container.upload_blob(
                name=f"{image_folder}{image_name}",
                data=image_bytes,
                overwrite=True,
                content_type="image/jpeg",
            )
            logger.info(f" Uploaded image {image_name} to blob")

        else:
            logger.warning(" Blob container not available — skipping blob logging")

        return {"class": predicted_class, "confidence": confidence}

    except Exception as e:
        logger.error(f" Error during prediction: {e}", exc_info=True)
        return {"error": str(e)}
