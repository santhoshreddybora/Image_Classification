## Brain MRI Classification with Azure ML Deployment
This project classifies Brain MRI images into different categories using a ResNet50 deep learning model.
It is fully integrated with Azure Machine Learning for training, experiment tracking (MLflow), and real-time inference deployment.

### Features
Data Ingestion from Azure Blob Storage
Data Augmentation with Albumentations
Model Training using TensorFlow + Keras (ResNet50)
Experiment Tracking with MLflow in Azure ML
Deployment as an Azure ML Managed Online Endpoint
Two Input Modes for Predictions:
JSON Base64 image string
Direct image file upload
Postman / Python client support for endpoint testing

### Architecture
flowchart LR
    A[Azure Blob Storage] -->|Dataset| B[Data Ingestion]
    B --> C[Data Augmentation]
    C --> D[Model Training - ResNet50]
    D --> E[MLflow Tracking in Azure ML]
    E --> F[Model Registration in Azure ML]
    F --> G[Managed Online Deployment]
    G -->|Image Input| H[Azure ML Endpoint]
    H --> I[Prediction JSON Output]

📂 Project Structure
Brain_Classification/
│
├── Components/
│   ├── data_ingestion.py
│   ├── data_augmentation.py
│   ├── base_model.py
│
├── Azure_deployment/
│   ├── score.py                 # Endpoint inference script
│   ├── Update_deployment.py     # Azure ML deployment update
│
├── config/
│   ├── params.yaml               # Configurations
│
├── main.py                       # Main pipeline
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
### Environment Setup
```
conda create -n brain_mri python=3.9
conda activate brain_mri
pip install -r requirements.txt
```
### Training the Model
```
python main.py
```
This will:
Download dataset from Azure Blob Storage
Perform augmentation
Train ResNet50 model
Save model to outputs/model/
Log model & artifacts to MLflow in Azure ML
☁️ Deploying the Model
```
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<SUB_ID>",
    resource_group_name="<RG_NAME>",
    workspace_name="<WS_NAME>"
)

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="brain-mri-endpoint",
    model="brain_classification_model_Restnet50:1",
    environment="tf-env-new:1",
    code_configuration=CodeConfiguration(
        code="./Azure_deployment",
        scoring_script="score.py"
    ),
    instance_type="Standard_D2as_v4",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```
🧪 Testing the Endpoint
JSON (Base64 image)

```
import requests, base64
ENDPOINT = "<your-endpoint-url>/score"
KEY = "<your-endpoint-key>"
with open("sample.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()
payload = {"image_base64": img_base64}
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
response = requests.post(ENDPOINT, headers=headers, json=payload)
print(response.json())
```
Multipart Form-Data (Image File Upload)
```
import requests

ENDPOINT = "<your-endpoint-url>/score"
KEY = "<your-endpoint-key>"

files = {"file": open("sample.jpg", "rb")}
headers = {"Authorization": f"Bearer {KEY}"}

response = requests.post(ENDPOINT, headers=headers, files=files)
print(response.json())
```

📊 Example Output
json
```
{
  "class": 1,
  "confidence": 0.973
}
```
🛠 Tech Stack
Python 3.9
TensorFlow / Keras
Albumentations
Azure Machine Learning
MLflow
Azure Blob Storage
Flask