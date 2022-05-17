@ECHO OFF 
:: This batch file will prepare and call the training script
mkdir -p Tensorflow/workspace/images/test
mkdir -p Tensorflow/workspace/images/train
mkdir -p Tensorflow/workspace/images/detect_tflite_res
:: Prepare training data
git clone https://github.com/nicknochnack/GenerateTFRecord Tensorflow/scripts
:: Create the train data
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
:: Create the test data
Create the test data
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
:: Run the model_configs script
python model_configs.py
:: Copy Model Config to Training Folder
cp Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config Tensorflow/workspace/models/my_ssd_mobnet
