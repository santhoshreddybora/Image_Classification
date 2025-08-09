from azure.storage.blob import BlobServiceClient
import os
import zipfile
from dotenv import load_dotenv
from pathlib import Path
import yaml
from logger import logging
from config import *
import glob
from entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
load_dotenv()
with open("./config/params.yaml", "r") as file:
    params = yaml.safe_load(file)['data_ingestion']
conn_str = os.getenv("AZURE_BLOB_CONN_STR")
if conn_str is None:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")

class DataIngestion:
    def __init__(self):
        with open("./config/params.yaml", "r") as file:
            self.params = yaml.safe_load(file)['data_ingestion']
    def download_and_extract_data(self,):
        logging.info('Downloading and extracting data from Azure Blob Storage')
        try:
            blob_service_client=BlobServiceClient.from_connection_string(conn_str=conn_str)
            container_name=params['container_name']
            blob_client= blob_service_client.get_blob_client(container=container_name, blob=params['blob_name'])
            
            os.makedirs(params['data_zip_file_path'], exist_ok=True)

            for blob in blob_service_client.get_container_client(container_name).list_blobs():
                if blob.name == params['blob_name']:
                    with open(os.path.join(params['data_zip_file_path'], blob.name), "wb") as download_file:
                        download_file.write(blob_client.download_blob().readall())
                    print(f"Downloaded {blob.name} to {params['data_zip_file_path']}")
            logging.info(f"Downloaded {params['blob_name']} to {params['data_zip_file_path']}")
        except Exception as e:
            logging.error(f"Error downloading or extracting data: {e}")   
            
    def unzip_data(self,):
        try:
            zip_file_path=os.path.join(params['data_zip_file_path'],params['blob_name'])
            os.makedirs(params['data_unzip_path'], exist_ok=True)
            with zipfile.ZipFile(zip_file_path,'r')as zip_ref:
                zip_ref.extractall(params['data_unzip_path'])
            print(f"Unzipped data to {params['data_unzip_path']}")
            logging.info(f"Unzipped data to {params['data_unzip_path']}")
        except Exception as e:
            logging.error(f"Error unzipping data: {e}")
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")
            self.download_and_extract_data()
            self.unzip_data()
            print(params['data_unzip_path'])
            dataingestionartifact= DataIngestionArtifact(
                Unzipped_data_path=params['data_unzip_path']
                )
            logging.info(f"Data ingestion artifact created with path: {dataingestionartifact.Unzipped_data_path}")
            return dataingestionartifact
        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise e




    
