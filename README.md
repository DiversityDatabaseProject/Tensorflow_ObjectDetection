# Pre-requisites (For Win10 Installation):
First of all, create your python venv, and install requirements.txt
```diff 
pip3 install -r requirements.txt
````

### Create Directories:
```
mkdir -p Tensorflow
```

### Cloning the TensorFlow model garden (TF2) to the folder 'models'
```
cd Tensorflow
git clone https://github.com/tensorflow/models
```

### Download and install protocol buffers

```
mkdir -p Tensorflow/protoc
cd Tensorflow/protoc
download https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protoc-3.20.1-win64.zip
unzip protoc-3.20.1-win64

cd Tensorflow\models\research
<protoc bin path>\protoc object_detection\protos\*.proto --python_out=.
copy object_detection\packages\tf2\setup.py setup.py
python setup.py build
python setup.py install

cd Tensorflow/models/research/slim && pip3 install -e . 
```

### Verify Installation

```
python Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py
```

should get the below results:
```
I0516 14:51:38.362038 59520 test_util.py:2373] time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
I0516 14:51:38.364032 59520 test_util.py:2373] time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 24 tests in 18.022s

OK (skipped=1)
```

download http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz

```
mkdir Tensorflow/workspace/pre-trained-models
mv ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.gz Tensorflow/workspace/pre-trained-models
cd Tensorflow/workspace/pre-trained-models && tar -zxvf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
```

### Training

Creating a TF record and cloning the script from a github repo
```
mkdir Tensorflow/scripts
git clone https://github.com/nicknochnack/GenerateTFRecord Tensorflow/scripts
```
Make sure to have teh test and train images in their corresponding folders:

### Create the train data
```
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
```

### Create the test data
```
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
```

### Copy Model Config to Training Folder
```
cp Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config Tensorflow/workspace/models/my_ssd_mobnet
```

### Copy our model configs (from the pipeline.config file) to the training forlder
```
copy Tensorflow\workspace\pre-trained-models\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\pipeline.config Tensorflow\workspace\models\my_ssd_mobnet
```

### Configure the training pipeline
```
python model_configs.py
```

## Train the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=50000
```

sample results:
```
INFO:tensorflow:Step 500 per-step time 0.390s
I0516 15:49:02.200201 55000 model_lib_v2.py:705] Step 500 per-step time 0.390s
INFO:tensorflow:{'Loss/classification_loss': 0.20576711,
 'Loss/localization_loss': 0.1371309,
 'Loss/regularization_loss': 0.15272368,
 'Loss/total_loss': 0.49562168,
 'learning_rate': 0.053333}
I0516 15:49:02.202197 55000 model_lib_v2.py:708] {'Loss/classification_loss': 0.20576711,
 'Loss/localization_loss': 0.1371309,
 'Loss/regularization_loss': 0.15272368,
 'Loss/total_loss': 0.49562168,
 'learning_rate': 0.053333}
```

## Evaluate the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet
```
### Test Image Detection
```
python detect_from_image.py
```

### Test Camera Detection
Note: for some reason, I did not see the detections from my camera.<br>
TODO: check the codes for issues
```
python detect_from_cam.py
```

## Save the Model (Freezing the Graph)
```
python Tensorflow\models\research\object_detection\exporter_main_v2.py  --input_type=image_tensor --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --trained_checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet --output_directory=Tensorflow\workspace\models\my_ssd_mobnet\export
```
## Convert to TFLite
```
python Tensorflow\models\research\object_detection\export_tflite_graph_tf2.py \
--input_shapes=4,640,640,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops
```

## TODO

- refactor codes
- remove unnecessary files / codes
- create a script for all the folder creations and downloading and copying of files, etc
