from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment,CodeConfiguration
from azure.identity import DefaultAzureCredential
import os
import logging

class UpdateDeploymnet:
    def __init__(self,):
        self.mlclient=MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv('SUBSCRIPTION_ID'),
            resource_group_name=os.getenv('RESOURCE_GROUP'),
            workspace_name=os.getenv('WORKSPACE_NAME')
        )
    
    def depolymentupdate(self,):
        deployment=ManagedOnlineDeployment(
            name="blue",
            endpoint_name="rn-end091401",
            model="brain_classification_model_Restnet50:4",
            environment="tf-env-new:2",
            code_configuration=CodeConfiguration(
                code="./Azure_deployment",  # folder path containing score.py
                scoring_script="score.py"
            ),
            instance_type="Standard_D2AS_V4",
            instance_count=1
        )

        update_deployment_result=self.mlclient.online_deployments.begin_create_or_update(deployment=deployment).result()
        logging.info(f"Deployment updated {update_deployment_result}")
