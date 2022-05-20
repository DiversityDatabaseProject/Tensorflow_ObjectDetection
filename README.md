# Tensorflow Object Detection Model Training, Inference and TFLite Conversion
This repository contains codes for generating, training and testing an Object Detection Model for detecting faces on images, using the pre-trained model, <a href="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz">ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8</a>. It also has a script for converting the saved model to Tensorflow Lite, and an inference test on the converted model. The TFLite model was deployed on iOS Devices in a <a href="https://github.com/DiversityDatabaseProject/face_detection_ios">separate repository<a>

# Installation Instructions

First off, make sure to create a virtual environment, and activate it.<br>
Make sure to have git installed.<br>
## Shell Scripts:
There are separate scripts for Windows and Ubuntu installations. The Windows Powershell scripts are under the win_scripts folder, and the Ubuntu bash scripts are under the ubuntu folder.<br>
Note: Shell scripts (.ps1 files) should be run on Windows Powershell.<br>
Before running the scripts (both ps1 and python scripts), make sure you are in the project root directory, "Tensorflow_ObjectDetection"<br>
The scripts should be run in the order they are written in this readme file.<br>
## Install Dependencies
To install dependencies, create workspace folders, download and build tensorflow files, etc.<br>
For Windows:
```
.\win_scripts\init.ps1
```
For Ubuntu:
```
./ubuntu/init.sh
```

To verify the installation:
```
python Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py
```

### Image Labeller
Run below script to configure, download and launch an image labeller.<br>
For Windows:
```
.\win_scripts\label_images.ps1
```
For Ubuntu:
```
./ubuntu/label_images.sh
```

### AWS S3 Multipart Uploader / Downloader
This python script contains upload / download functions to and from S3 Bucket folders<br>
Note: Below script will upload all files in the local folder to the S3 Bucket folder.
```
python s3_utils.py --opt upload --from test --to bucket/folder
```
Note: Below script will download all files from S3 Bucket folder to the local folder.
```
python s3_utils.py --opt download --from bucket/folder --to test
```

### Model Training
First, make sure to copy your annotated test and train images in the corresponding folders.<br>
```
Tensorflow\workspace\images\test
Tensorflow\workspace\images\train
```
If everything is OK, you can launch the train_configs.bat<br>
For Windows:
```
.\win_scripts\train_configs.ps1
```
For Ubuntu:
```
./ubuntu/train_configs.sh
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
Make sure that the workspace folder and test images exist before running the script.
```
python detect_from_image.py  --checkpoint Tensorflow\workspace\models\my_ssd_mobnet\ckpt-51 --label_map Tensorflow\workspace\annotations\label_map.pbtxt --threshold .5 --images_folder Tensorflow\workspace\images\detect_image --output_path Tensorflow\workspace\images\detect_res
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
python detect_image_tflite.py --tf_model Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\detect_cl.tflite  --tf_labels Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\labels.txt --threshold .5 --images_folder Tensorflow\workspace\images\detect_image --output_path Tensorflow\workspace\images\detect_tflite_res
```

### Resources
<a href="https://github.com/nicknochnack/TFODCourse">Full Object Detection Course Github Repo</a><br>
<a href="https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API">Tensorflow Lite Object Detection API Github Repo</a><br>
<a href="https://www.tensorflow.org/lite/convert">TFLite Conversion Article</a><br>
<a href="https://www.tensorflow.org/lite/examples/object_detection/overview"> TFLite Object Detection Article</a><br>
<a href="https://github.com/tzutalin/labelImg">Image Labeller</a>

## TODO

- include default values to python parameters, if none given by user
- test the bash and powershell scripts
- add error handling in the scripts