from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers,models
import yaml
import os
from logger import logging
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
import numpy as np
from entity.artifact_entity import DataAugmentationArtifact
import mlflow
from azureml.core import Workspace
from tensorflow.keras.models import load_model
from entity.artifact_entity import ModelArtifact
import mlflow.keras
import tensorflow as tf 
class ModelBuilder:
    def __init__(self,config_path='config/params.yaml'):
        with open(config_path,'r') as file:
            self.config=yaml.safe_load(file)['model_builder']
            try:
                # Connect to Azure ML workspace
                ws = Workspace(
                    subscription_id=os.getenv("SUBSCRIPTION_ID"),
                    resource_group=os.getenv('RESOURCE_GROUP'),
                    workspace_name=os.getenv('WORKSPACE_NAME')
                )

                # Set MLflow tracking URI from workspace
                tracking_uri = ws.get_mlflow_tracking_uri()
                mlflow.set_tracking_uri(tracking_uri)
                logging.info(f"MLflow tracking URI set to: {tracking_uri}")
            except Exception as e:
                logging.warning(f"Failed to set MLflow tracking URI from Azure ML: {e}")

    def build_model(self):
        try :
            logging.info("Building the model")
            input_tensor = tf.keras.Input(shape=(224, 224, 3))
            base_model=ResNet50(weights=self.config['weights'], include_top=False, input_tensor=input_tensor)
            for layer in base_model.layers:
                layer.trainable = False
            
            model=models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(self.config['dropout_rate']),
                layers.Dense(self.config['dense_units'], activation='relu'),
                layers.Dropout(self.config['dropout_rate']),
                layers.Dense(self.config['num_classes'], activation='softmax')
            ])

            logging.info("Model built successfully")
            logging.info(f"Model summary: {model.summary()}")
            return base_model,model
        except Exception as e:
            logging.error(f"Error building the model: {e}")
            raise e
    
    def compile_and_fit(self, model, train_dataset, val_dataset, class_weights):
        try:
            logging.info("Compiling the model")
            model.compile(optimizer=self.config['optimizer'],
                          loss=self.config['loss'],
                          metrics=self.config['metrics'])
            
            logging.info("Fitting the model")
            history = model.fit(train_dataset,
                                validation_data=val_dataset,
                                epochs=self.config['epochs'],
                                class_weight= class_weights)
            logging.info("Model training completed")
            return model,history
        except Exception as e:
            logging.error(f"Error during model compilation or fitting: {e}")
            raise e
    def transfer_learning(self, model, train_dataset, val_dataset,class_weights):
        try:
            logging.info("Starting transfer learning")
            base_model=model.layers[0]  # Get the base model from the Sequential model
            for layer in base_model.layers[:-20]:
                layer.trainable = False  # Freeze all but last 20

            for layer in base_model.layers[-20:]:
                layer.trainable = True   # Fine-tune last 20 layers
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['transfer_learning_lr'])

            model.compile(optimizer=optimizer,
                          loss=self.config['loss'],
                          metrics=self.config['metrics'])
            logging.info("Compiling the model for transfer learning")
            logging.info(f"Model summary after transfer learning:{model.summary()}")
            logging.info("Fitting the model for transfer learning")
            history = model.fit(train_dataset,
                                validation_data=val_dataset,
                                epochs=self.config['transfer_epochs'],
                                class_weight=class_weights)
            
            
            logging.info("Transfer learning completed")
            return model, history
        except Exception as e:
            logging.error(f"Error during transfer learning: {e}")
            raise e
    
    def test_model(self,model,test_dataset):
        try:
            os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "900"  # 10 mins
            logging.info("Testing the model")
            test_loss, test_accuracy = model.evaluate(test_dataset)
            logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
            y_true=[]
            y_pred=[]
            y_scores=[]
            for imgs, labels in test_dataset:
                preds = model.predict(imgs)  # shape (batch_size, num_classes) or (batch_size,)
                # Convert probabilities/logits to discrete class predictions
                class_preds = np.argmax(preds, axis=1)  # for multi-class
                # For binary classification: class_preds = (preds > 0.5).astype(int).flatten()
                y_true.append(labels.numpy())
                y_pred.append(class_preds)
                y_scores.append(preds)
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            y_scores = np.concatenate(y_scores)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)
            roc_auc=roc_auc_score(y_true, y_scores,multi_class='ovr', average='macro')
            try:
                os.makedirs(os.path.dirname(self.config['keras_model_path']), exist_ok=True)
                # model.save(self.config['model_save_path'])
                mlflow.keras.save_model(model,path=self.config['keras_model_path'])
                logging.info(f"Model save success in {self.config['keras_model_path']}")
            except Exception as e:
                logging.error(f"Error saving the model: {e}")
            mlflow.set_experiment('Resnet50_Model_Training')
            with mlflow.start_run(run_name='Resnet50_Model_Training') as run:
                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc_score", roc_auc)
                mlflow.log_artifact(self.config['model_config_file_path'], artifact_path='config')
                # mlflow.log_artifact(self.config['keras_model_path'], artifact_path="models")
                # mlflow.tensorflow.log_model(model,artifact_path='models')
                mlflow.keras.log_model(model, artifact_path='model') ## works with low version of mlflow
                logging.info("metrics and artifacts logged to azure ML flow")

            
            run_id = run.info.run_id
            logging.info(f"Logged to run_id:{run_id}")
            logging.info(f"Accuracy Score: {acc}")
            logging.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Precision: {prec}")
            logging.info(f"ROC AUC Score: {roc_auc}")
            logging.info(f"Recall: {rec}")
            logging.info("Model testing completed")
            return test_loss, test_accuracy,run_id
        except Exception as e:
            logging.error(f"Error during model testing: {e}")
            raise e
    def initiate_model_building(self, train_dataset, val_dataset,class_weights):
        try:
            # logging.info("Initiating model building process")
            # base_model,model = self.build_model()
            # model, history = self.compile_and_fit(model, train_dataset, val_dataset, class_weights)
            # model, history = self.transfer_learning(model, train_dataset, val_dataset,class_weights)
            model=load_model('outputs/data/model')
            test_loss,test_accuracy,run_id=self.test_model(model, val_dataset)
            logging.info("Model building process completed")
            model_artifact = ModelArtifact(
                model=model,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                model_save_path=self.config['model_save_path'],
                keras_model_save_path=self.config['keras_model_path'],
                model_config_file_path=self.config['model_config_file_path'],
                mlflow_experiment_name=self.config['mlflow_experiment_name'],
                run_id=run_id
            )
            return model_artifact
        except Exception as e:
            logging.error(f"Error in initiate_model_building: {e}")
            raise e
    
            

