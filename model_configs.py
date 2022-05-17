import os
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import urllib.request
import tarfile


#cluster model folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

#setting up the name of our pre train model, this include the name of the file from the pre train TensorFlow 2 Detection Model Zoo
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'


 #setting up variables
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

#creating our path for working folder
paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'), # where TF record file will be stored
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'), # folder for the selected models tested
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'UNZIPPED_MODEL_NAME': os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME)
}

# Download pre-trained model
urllib.request.urlretrieve(PRETRAINED_MODEL_URL, files['UNZIPPED_MODEL_NAME'])

my_tar = tarfile.open(files['UNZIPPED_MODEL_NAME'])
my_tar.extractall(paths['PRETRAINED_MODEL_PATH']) # specify which folder to extract to
my_tar.close()

#Create directories if not existing
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.makedirs(path)
        if os.name == 'nt':
            os.makedirs(path)

def create_label_map():
    #creating a map to the label in a text file 
    label_list = [{'name':'face', 'id':1}]

    with open(files['LABELMAP'], 'w') as f:
        for label in label_list:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def create_pipeline_config():
    #Using protocol buffer to process the config file

    #Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.
    #Args: pipeline_config_path: Path to pipeline_pb2.TrainEvalPipeline Config text proto.
    #config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override pipeline_config_path.

    #Script from: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/config_util.py 
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'],PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

if __name__ == '__main__':
    create_label_map()
    create_pipeline_config()