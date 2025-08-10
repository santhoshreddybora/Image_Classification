import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

if __name__ == "__main__":
    # Azure ML batch passes input and output folders as command-line args
    input_dir = sys.argv[1]   # Directory containing input images
    output_dir = sys.argv[2]  # Directory to write predictions

    # Load model from Azure ML mounted model directory
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model")
    model = load_model(model_path)

    results = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        try:
            input_array = preprocess_image(file_path)
            preds = model.predict(input_array)
            predicted_class = int(np.argmax(preds))
            confidence = float(np.max(preds))
            results.append({
                "filename": filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "error": str(e)
            })

    # Save predictions as CSV in output directory
    output_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
