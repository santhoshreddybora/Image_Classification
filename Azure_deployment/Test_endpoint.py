import requests,base64,json
from pathlib import Path
import sys
import os
Endpoint_URL ="https://rn-end091401.eastus.inference.ml.azure.com/score"
KEY="3pq66ugB26tGpkjOm6Gb4DoQ87wHGuoFqXCNO3JwZz13Pts8206TJQQJ99BHAAAAAAAAAAAAINFRAZML4Y2r"

HEADERS_JSON = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}"
}
HEADERS_FILE = {
    "Authorization": f"Bearer {KEY}"
}

def predict_with_base64(image_path):
    """Send request with JSON Base64 encoded image"""
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    payload = {"image_base64": img_base64}
    response = requests.post(Endpoint_URL, headers=HEADERS_JSON, json=payload)
    return response.json()

def predict_with_file(image_path):
    """Send request with multipart/form-data"""
    files = {"file": open(image_path, "rb")}
    response = requests.post(Endpoint_URL, headers=HEADERS_FILE, files=files)
    return response.json()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_endpoint.py <method> <image_path>")
        print("method: json or file")
        sys.exit(1)

    method = sys.argv[1].lower()
    image_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    if method == "json":
        result = predict_with_base64(image_path)
    elif method == "file":
        result = predict_with_file(image_path)
    else:
        print("Invalid method. Use 'json' or 'file'.")
        sys.exit(1)

    print("Prediction Result:", result)

