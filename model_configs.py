'''
This python script contains configuration settings for model training,
creating the class label (label_map.pbtxt)
and creating the pipeline config file (pipeline.config)
Refactored from a code on Jupyter notebook by Maria Rosario SEBASTIAN, May 2022
Source: https://github.com/nicknochnack/TFODCourse
'''
import os
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from google.protobuf import text_format
import urllib.request
import tarfile
import load_configs as cf

def setup_pretrained_model():
    ''' Set up pre*trained model paths '''
    # Download pre-trained model
    urllib.request.urlretrieve(cf.PRETRAINED_MODEL_URL, cf.files['UNZIPPED_MODEL_NAME'])

    my_tar = tarfile.open(cf.files['UNZIPPED_MODEL_NAME'])

    # Create extract folder
    if not os.path.exists(cf.paths['PRETRAINED_MODEL_PATH']):
        os.makedirs(cf.paths['PRETRAINED_MODEL_PATH'])

    # Extract to folder
    my_tar.extractall(cf.paths['PRETRAINED_MODEL_PATH'])
    my_tar.close()


def create_label_map():
    ''' creating a map to the label in a text file ''' 
    label_list = [{'name':'face', 'id':1}]

    with open(cf.files['LABELMAP'], 'w') as f:
        for label in label_list:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def create_pipeline_config():
    ''' Using protocol buffer to process the config file

    Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.
    Args: pipeline_config_path: Path to pipeline_pb2.TrainEvalPipeline Config text proto.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override pipeline_config_path.

    Script from: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/config_util.py '''
    # Create detection results folder
    if not os.path.exists(cf.paths['CHECKPOINT_PATH']):
        os.makedirs(cf.paths['CHECKPOINT_PATH'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(cf.files['PRE_TRAINED_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(cf.paths['PRETRAINED_MODEL_PATH'],cf.PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= cf.files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(cf.paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = cf.files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(cf.paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(cf.files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)
    print(cf.files['PIPELINE_CONFIG'])
    
if __name__ == '__main__':
    setup_pretrained_model()
    create_label_map()
    create_pipeline_config()