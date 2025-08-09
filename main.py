from Components.data_ingestion import DataIngestion
from entity.artifact_entity import DataIngestionArtifact
from Components.data_Augmentation import DataAugmentation
from logger import logging
from Components.base_model import ModelBuilder
from Azure_deployment.Endpoint_deployment import AzureDeployment
from Azure_deployment.Update_deployment import UpdateDeploymnet
if __name__ == "__main__":
    try:
        logging.info('starting the pipeline')
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
        data_augmentation = DataAugmentation(data_ingestion_artifact=data_ingestion_artifact)
        artifact = data_augmentation.initiate_data_augmentation()
        train_dataset = artifact.train_dataset
        val_dataset = artifact.val_dataset
        compute_class_weights_dict = artifact.compute_class_weights_dict
        logging.info(f"Data augmentation artifact: {artifact}")
        logging.info(f"Train dataset: {train_dataset}, Validation dataset: {val_dataset}")
        model_builder = ModelBuilder()
        model_artifact= model_builder.initiate_model_building(train_dataset=train_dataset,
                                                              val_dataset=val_dataset,
                                                              class_weights=compute_class_weights_dict)
        logging.info(f"Model: {model_artifact.model}")
        azuredeploy=AzureDeployment()
        scoring_uri,scoring_key=azuredeploy.initalize_deployment(run_id=model_artifact.run_id)
        # depolyment_updt= UpdateDeploymnet()  -- if you want to update the deployment we can use this class
        # depolyment_updt.depolymentupdate()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e
