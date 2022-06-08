'''
This python script contains constants and function that are used by the other codes.
Author: Maria Rosario SEBASTIAN
Date: May 2022
'''
import os
import configparser
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

config = configparser.ConfigParser()
config.read('utils/config.ini')

#From config file
#setting up the name of our pre train model, this include the name of the file from the pre train TensorFlow 2 Detection Model Zoo
PRETRAINED_MODEL_NAME = config['MODELS']['PRETRAINED_MODEL_NAME']
UNZIPPED_MODEL = config['MODELS']['UNZIPPED_MODEL']
PRETRAINED_MODEL_URL = config['MODELS']['PRETRAINED_MODEL_URL']
LABEL_MAP_NAME=config['TRAIN']['LABEL_MAP_NAME']

#cluster model folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
DETECTED_IMAGE_NAME = 'detection_test.png'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'), # high level workspace
    'SCRIPTS_PATH': os.path.join('Tensorflow','GenerateTFRecord'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': config['TRAIN']['TRAIN_ANNOTATIONS'], # where TF record file will be stored
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'), # folder for the selected models tested
    'PRETRAINED_MODEL_PATH': config['MODELS']['PRE_TRAINED_MODELS_PATH'],
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'DETECT_RES_PATH': os.path.join('Tensorflow', 'workspace','images','detect_res'),
    'TEST_IMAGE_PATH' : os.path.join('Tensorflow', 'workspace','images', 'detect_image')
 }

files = {
    'PIPELINE_CONFIG': config['TRAIN']['PIPELINE_CONFIG'],
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'DETECTED_IMAGE': os.path.join(paths['DETECT_RES_PATH'], DETECTED_IMAGE_NAME),
    'UNZIPPED_MODEL_NAME': os.path.join(paths['PRETRAINED_MODEL_PATH'], UNZIPPED_MODEL)
}

#Create directories if not existing
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.makedirs(path)
        if os.name == 'nt':
            os.makedirs(path)

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections