# Pre-requisites (For Win10 Installation):
First off, make sure to create a virtual environment, and activate it.<br>
Make sure to have git installed.<br>
Note: Shell scripts (.ps1 files) should be run on Windows Powershell.<br>
Before running the scripts (both ps1 and python scripts), make sure you are in the project root directory, "Tensorflow_ObjectDetection"<br>
The scripts should be run in the order they are written in this readme file.<br>
## Install Dependencies
To install dependencies, create workspace folders, download and build tensorflow files, etc:
```
.\win_scripts\init.ps1
```

To verify the installation:
```
python Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py
```

### Image Labeller
Run below script to configure, download and launch an image labeller.<br>
```
.\win_scripts\label_images.ps1
```

### Model Training
First, make sure to copy your annotated test and train images in the corresponding folders.<br>
```
Tensorflow\workspace\images\test
Tensorflow\workspace\images\train
```
If everything is OK, you can launch the train_configs.bat
```
.\win_scripts\train_configs.ps1
```

## Train the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=50000
```

## Evaluate the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet
```

### Test Image Detection (Inference test)
Before running this test, make sure there are test images in the folder:
```
Tensorflow\workspace\images\detect_image
```
This script will make an inference on the images in the above folder and save the results in this folder:
```
Tensorflow\workspace\images\detect_image\detect_res
```
Below script will test image detection with bounding box and score for a given image. <br>
Note that checkpoint_path should contain all the files that are set in the parameters (label_map, checkpoint), including the saved model. In our repository, this should be under: Tensorflow\workspace\models\my_ssd_mobnet
```
python detect_from_image.py --checkpoint_path path\to\checkpoint --checkpoint_name ckpt-51 --label_map label_map.pbtxt --threshold .5 --images_folder path\to\image\folder --output_path path\to\inference\results\folder
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
python Tensorflow\models\research\object_detection\export_tflite_graph_tf2.py --input_shapes=4,640,640,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops
```

## Test TFLite converted model (inference test)
Before running this test, make sure there are test images in the folder:
```
Tensorflow\workspace\images\detect_image
```
This script will make an inference on the images in the above folder and save the results in this folder:
```
Tensorflow\workspace\images\detect_image\detect_tflite_res
```
Run below script to make the inference test:
```
python detect_image_tflite.py --tf_model path\to\models\tflitemodelname.tflite  --tf_labels path\to\labelfile\labels.txt --threshold .5 --images_folder path\to\image\folder --output_path path\to\inference esults\folder
```

### Resources
<a href="https://github.com/nicknochnack/TFODCourse">Full Object Detection Course Github Repo</a><br>
<a href="https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API">Tensorflow Lite Object Detection API Github Repo</a><br>
<a href="https://www.tensorflow.org/lite/convert">TFLite Conversion Article</a><br>
<a href="https://www.tensorflow.org/lite/examples/object_detection/overview"> TFLite Object Detection Article</a><br>
<a href="https://github.com/tzutalin/labelImg">Image Labeller</a>

## TODO

- refactor codes
- test the scripts
- remove unnecessary files / codes
