from dataclasses import dataclass
import tensorflow as tf
import mlflow.keras 
@dataclass
class DataIngestionArtifact:
    Unzipped_data_path: str


@dataclass
class DataAugmentationArtifact:
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    compute_class_weights_dict: dict

@dataclass
class ModelArtifact:
    model: tf.keras.Model
    test_loss: float
    test_accuracy: float
    run_id: str
    model_save_path: str 
    keras_model_save_path: str
    model_config_file_path: str 
    mlflow_experiment_name: str 

@dataclass
class Deployment:
    scoring_uri: str
    scoring_key: str
