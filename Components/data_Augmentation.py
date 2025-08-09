import albumentations as A
import os
import cv2
import numpy as np
from logger import logging
import yaml
from sklearn.model_selection import train_test_split
from entity.artifact_entity import DataIngestionArtifact
from entity.artifact_entity import DataAugmentationArtifact
import glob
import pydicom
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

class DataAugmentation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,config_path='config/params.yaml'):
        with open(config_path,'r') as file:
            self.config=yaml.safe_load(file)['data_augmentation']
        self.data_ingestion_artifact = data_ingestion_artifact
        self.train_aug=A.Compose([
                        A.RandomResizedCrop(224,224,scale=(0.8,1.0),p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.Rotate(limit=15, p=0.5),
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                        A.Resize(224, 224), 
                        A.Normalize() ])
        self.val_aug=A.Compose( [
                                A.Resize(224, 224),
                                A.Normalize()
                                ])

    def all_files_and_labels(self,unzipped_data_path):
        all_files=[]
        labels=[]
        classes = ['brain_glioma', 'brain_menin', 'brain_tumor']
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        unzipped_data_path = os.path.join(unzipped_data_path, 'Brain_Cancer')
        logging.info(f"Unzipped data path: {unzipped_data_path}")
        try:
            for cls in classes:
                class_dir=os.path.join(unzipped_data_path, cls)
                img_paths=glob.glob(os.path.join(class_dir,'*'))
                all_files.extend(img_paths)
                labels.extend([class_to_index[cls]] * len(img_paths))
                X_train,X_test,y_train,y_test = train_test_split(all_files, labels, test_size=self.config['test_size'], random_state=self.config['random_state'])
            logging.info(f"Found {len(all_files)} images across {len(classes)} classes.")
        except Exception as e:
            logging.error(f"Error processing files and labels: {e}")
        return X_train, X_test, y_train, y_test
    
    def read_image(self, image_path):
        ext=os.path.splitext(image_path)[-1].lower()
        try:
            if ext == '.dcm':
                ds = pydicom.dcmread(image_path)
                image = ds.pixel_array
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3 and image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    logging.error(f"Unexpected image shape: {image.shape} for DICOM file {image_path}")
            else:
                image = cv2.imread(image_path)
                if image is None:
                    logging.info(f"Image at {image_path} could not be read.")
                elif len(image.shape) == 3 and image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:  # Grayscale image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    logging.error(f"Unexpected image shape: {image.shape} for file {image_path}")               
            return image
        except Exception as e:
            logging.error(f"Error reading image {image_path}: {e}")

        
    def preprocess_image(self, path_str, label, augment=True):
        try:
            img = self.read_image(path_str)
            aug = self.train_aug if augment else self.val_aug
            img = aug(image=img)['image']
            return img.astype(np.float32), np.int64(label)
        except Exception as e:
            logging.error(f"Error in preprocessing image {path_str}: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32), np.int64(label)  # fallback image


    def tf_augment(self, augment=True):
        def wrapper(path, label):
            def _pyfunc_wrapper(p, l):
                decoded_path = p.numpy().decode("utf-8")
                return self.preprocess_image(decoded_path, l, augment)

            img, label = tf.py_function(
                _pyfunc_wrapper,
                [path, label],
                [tf.float32, tf.int64]
            )
            img.set_shape([224, 224, 3])
            label.set_shape([])
            logging.info(f"Processed image and labels")
            return img, label

        return wrapper
        
    def create_datasets(self,paths,labels,batch_size,augment=True,shuffle=True):
        try:
            logging.info('''Starting data augmentation process''')
            df=tf.data.Dataset.from_tensor_slices((paths,labels))
            if shuffle:
                df=df.shuffle(buffer_size=len(paths))
            df=df.map(self.tf_augment(augment), num_parallel_calls=tf.data.AUTOTUNE)
            return df.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        except Exception as e:
            logging.error(f"Error in create_datasets: {e}")
            raise e
    
    def initiate_data_augmentation(self):
        logging.info("Initiating data augmentation process")
        try:
            unzipped_data_path=self.data_ingestion_artifact.Unzipped_data_path
            logging.info(f"Unzipped data path: {unzipped_data_path}")
            X_train,X_test,y_train,y_test=self.all_files_and_labels(unzipped_data_path)
            compute_class_weights= compute_class_weight(
                class_weight='balanced',y=y_train,classes=np.unique(y_train))
            logging.info(f"Computed class weights: {compute_class_weights}")
            compute_class_weights_dict = dict(enumerate(compute_class_weights))
            logging.info(f"Class weights: {compute_class_weights_dict}")
            logging.info(f"Train files: {len(X_train)}, Test files: {len(X_test)}")
            train_dataset=self.create_datasets(X_train,y_train, 
                                               batch_size=self.config['batch_size'], 
                                               augment=True, 
                                               shuffle=True)
            val_dataset=self.create_datasets(X_test,y_test, 
                                             batch_size=self.config['batch_size'], 
                                             augment=False, 
                                             shuffle=False)
            logging.info(f"Train dataset: {train_dataset}, Validation dataset: {val_dataset}")
            logging.info(enumerate(train_dataset))
            data_augmentation_artifact=DataAugmentationArtifact(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                compute_class_weights_dict=compute_class_weights_dict
            )
            logging.info(f"Data augmentation artifact created with train and validation datasets.")
            return data_augmentation_artifact
        except Exception as e:
            logging.error(f"Error in initiate_data_augmentation: {e}")
            raise e
            return jsonify({'error': str(e)}), 500
    