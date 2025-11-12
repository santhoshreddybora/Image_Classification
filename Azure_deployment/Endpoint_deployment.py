
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.entities import Model,Environment,CodeConfiguration
from azure.ai.ml.constants import AssetTypes
import yaml
from dotenv import load_dotenv
from logger import logging
load_dotenv()
import os
# from entity.artifact_entity import Deployment
from datetime import datetime

class AzureDeployment:
    def __init__(self,):
        self.ml_client=MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("RESOURCE_GROUP"),
        workspace_name=os.getenv("WORKSPACE_NAME")
    )

# run_id="74d052cc-a937-410a-a234-b6adb8c1fe80"
# model_path = f"azureml://jobs/{run_id}/outputs/models"
    def register_model(self, runid: str):
            try:
                if not runid:
                    raise ValueError("Run id is required to register the model")

                # Match the artifact_path used in mlflow.keras.log_model
                model_path = f"runs:/{runid}/model"  

                registered_model = self.ml_client.models.create_or_update(
                    Model(
                        name="brain_classification_model_Restnet50",
                        path=model_path,
                        description="Brain Classification Model using ResNet50",
                        type=AssetTypes.MLFLOW_MODEL,   # <-- since you logged with mlflow.keras.log_model
                        tags={"framework": "TensorFlow", "version": "2.12"}
                    )
                )

                logging.info(
                    f" Registered model: {registered_model.name}, version: {registered_model.version}"
                )
                return registered_model
            except Exception as e:
                logging.error(f" Error occurred while registering the model: {e}")
                raise e
    
    def deployments(self,model):
        try:
            # create env
            env=Environment(
                name="tf-env-new",
                description="tensorflow Inference Environment",
                conda_file='Azure_deployment/env.yaml',
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
                # inference_config=True
            )
            self.ml_client.environments.create_or_update(env)
            #Create Endpoint
            endpoint_name="rn-end"+ datetime.now().strftime("%d%H%M")
            endpoint=ManagedOnlineEndpoint(
                name=endpoint_name,
                description="Real-time endpoint for RestNet50",
                auth_mode="key"
            ) 
            endpoint_result=self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logging.info(f"Endpoint created {endpoint_result}")
            # endpoint=self.ml_client.online_endpoints.get(name=endpoint_name)
            scoringuri=endpoint_result.scoring_uri
            logging.info(f"endpoint scoring uri{scoringuri}")
            keys = self.ml_client.online_endpoints.get_keys(name=endpoint_name)
            scoring_key=keys.primary_key
            logging.info(f"Primary key {scoring_key}")
            # create Deployment
            deployment=ManagedOnlineDeployment(
                name='blue',
                endpoint_name=endpoint_name,
                model=model,
                environment=env,
                code_configuration=CodeConfiguration(
                    code="./Azure_deployment",
                    scoring_script="score.py"
                ),
                instance_type="Standard_D2AS_V4",
                instance_count=1
            )
            deployment_result=self.ml_client.online_deployments.begin_create_or_update(deployment).result()
            logging.info(f"Model deployed successfully. Deployment Reuslt{deployment_result}")

            endpoint_traffic=ManagedOnlineEndpoint(name=endpoint_name,
                                                   traffic={"blue":100}
                                                   )

            updated_endpoint=self.ml_client.online_endpoints.begin_create_or_update(endpoint_traffic).result()
            logging.info(f"Traffic updated 100% to blue model. {updated_endpoint}")
            return scoringuri,scoring_key
        except Exception as e:
            logging.info(f"Error occurred when creating as endpoint and deployment {e}")
        
        

    def initalize_deployment(self,run_id):
        try:
            logging.info(f"Initialize the deployment in azure with runid:{run_id}")
            resgistered_model =self.register_model(run_id)
            logging.info(f"Registered model version: {resgistered_model.version}")
            model=self.ml_client.models.get(name="brain_classification_model_Restnet50",version=resgistered_model.version)
            scoringuri,scoring_key=self.deployments(model)
            return scoringuri,scoring_key
        except Exception as e:
            logging.info(f"Error occurred while initiating the deployment{e}")
