'''
This python script loads config file values.
Author: Maria Rosario SEBASTIAN
Date: May 2022
'''
import os
import configparser

config = configparser.ConfigParser()
config.read('utils/config.ini')

#From config file
#setting up the name of our pre train model, this include the name of the file from the pre train TensorFlow 2 Detection Model Zoo
PRETRAINED_MODEL_NAME = config['MODELS']['PRETRAINED_MODEL_NAME']
UNZIPPED_MODEL = config['MODELS']['UNZIPPED_MODEL']
PRETRAINED_MODEL_URL = config['MODELS']['PRETRAINED_MODEL_URL']
LABEL_MAP_NAME=config['TRAIN']['LABEL_MAP_NAME']
SAVED_MODEL_DIR = config['MODELS']['SAVED_MODEL_DIR']
TFLITE_MODEL_QUANTIZED = config['MODELS']['TFLITE_MODEL_QUANTIZED']
SAVED_MODEL_UPDATED = config['MODELS']['SAVED_MODEL_UPDATED'] 

#cluster model folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
DETECTED_IMAGE_NAME = 'detection_test.png'

#apps
IMAGE_UPLOAD=config['APP']['IMAGE_UPLOAD']
IMAGE_UPLOAD_RES=config['APP']['IMAGE_UPLOAD_RES']

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
    'PRE_TRAINED_CONFIG': os.path.join(config['MODELS']['MODEL_FPN_PATH'],'pipeline.config'),
    'PIPELINE_CONFIG': config['TRAIN']['PIPELINE_CONFIG'],
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'DETECTED_IMAGE': os.path.join(paths['DETECT_RES_PATH'], DETECTED_IMAGE_NAME),
    'UNZIPPED_MODEL_NAME': os.path.join(paths['PRETRAINED_MODEL_PATH'], UNZIPPED_MODEL)
}
