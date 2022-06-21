# Tensorflow Object Detection Model Training, Inference and TFLite Conversion
This repository contains codes for generating, training and testing an Object Detection Model for detecting faces on images, using the pre-trained model, <a href="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz">ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8</a>. It also has a script for converting the saved model to Tensorflow Lite, and an inference test on the converted model. The TFLite model was deployed on iOS Devices in a <a href="https://github.com/DiversityDatabaseProject/face_detection_ios">separate repository<a>

## Pre-requisites
- Create Python3.x virtual environmentand and activate it. I STRONGLY advise against 'conda', at least NOT for Windows. It will mess up your life. And give you nightmares. And lots of brain cells dying for nothing.<br>
- git<br>

## About the Shell Scripts
This project contains shell scripts for installation on Windows and Ubuntu.<br>
Windows Powershell (.ps1 files) should be run on Windows Powershell, not on the old DOS-like command prompt.<br><br>
For Ubuntu, there will be cases when there will be strange characters added at the end of each line in the ubuntu scripts as a consequence of the transformation from Windows to Linux.<br>
To get rid of this, you can run the below script on all the .sh files to remove the extra characters:
```
sed -i 's/\r//g' <replace_this_with_your_script.sh>
```
Before running the scripts (both shell scripts and python scripts), make sure you are in the project root directory, "Tensorflow_ObjectDetection"<br>
The scripts should be run in the order they are written in this readme file.<br>
  
## SET UP THE WORKSPACE, TRAIN, TEST, CONVERT TO TFLITE
### 1. Install Dependencies
To install dependencies, create workspace folders, download and build tensorflow files, etc.<br>
For Windows:
```
.\win_scripts\init.ps1
```
For Ubuntu:
```
chmod -R 777 ./ubuntu
./ubuntu/init.sh
```
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_init_screenshot.PNG"/>

### Possible Issue
If there are issues on requirements.txt installations and some python packages were not installed, just run pip install again:
```
pip3 install -r requirements.txt
```

To verify the installation:
```
python Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_verify_installation.PNG"/>
  
### 2. Model Training set up and configs
Before doing this step, make sure your annotated test and train images are ready. You can use the <a href="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/edit/main/README.md#image-labeller">image labeling tool</a> to label the images.<br>
Make sure to copy your annotated test and train images in the corresponding folders.<br>
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
 
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_train_configs.PNG"/>

### 3. Train the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=50000
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_train.PNG"/>

### 4. Evaluate the model
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_evaluation.PNG"/>

### 5. Save the Model (Freezing the Graph)
```
python Tensorflow\models\research\object_detection\export_tflite_graph_tf2.py  --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --trained_checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet --output_directory=Tensorflow\workspace\models\my_ssd_mobnet\export
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_saved_model.PNG"/>
  
### 6. Test Image Detection (Inference test)
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

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_detect_res.PNG"/>
  
### 7. Test Camera Detection
Note: for some reason, I did not see the detections from my camera.<br>
TODO: check the codes for issues
```
python detect_from_cam.py
```
Press 'q' to quit.

<img width="640" height="480" src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_live_cam_detection.PNG"/>

### 8. Convert to TFLite (commandline)

```
tflite_convert --saved_model_dir=Tensorflow\workspace\models\my_ssd_mobnet\export\saved_model --output_file=Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\detect.tflite --input_shapes=4,640,640,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops
```

Copy labels.txt to tfliteexport folder:
```
cp .\utils\labels.txt .\Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_tflite_conversion.PNG"/>

### 9. Test TFLite converted model (inference test)
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
python detect_image_tflite.py --tf_model Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\detect.tflite  --tf_labels Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\labels.txt --threshold .5 --images_folder Tensorflow\workspace\images\detect_image --output_path Tensorflow\workspace\images\detect_tflite_res
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_detect_res_tflite.PNG"/>

### 10. TFLite Quantization
<a href="https://www.tensorflow.org/lite/performance/post_training_quantization">Post-training quantization</a> is a conversion technique that can reduce model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy. You can quantize an already-trained float TensorFlow model when you convert it to TensorFlow Lite format using the TensorFlow Lite Converter.

```
python convert_tflite_quantized.py
```

<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_detect_tflite_quantization.PNG"/>

### 9. Test TFLite quantized model (inference test)
Before running this test, make sure there are test images in the folder:
```
Tensorflow\workspace\images\detect_image
```

This script will make an inference on the images in the above folder and save the results in this folder:
```
Tensorflow\workspace\images\detect_image\detect_tflite_quantized_res
```
Run below script to make the inference test:
```
python detect_image_tflite.py --tf_model Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\detect_quantized.tflite  --tf_labels Tensorflow\workspace\models\my_ssd_mobnet\tfliteexport\labels.txt --threshold .5 --images_folder Tensorflow\workspace\images\detect_image --output_path Tensorflow\workspace\images\detect_tflite_quantized_res
```
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/win_detect_res_tflite_quantized.PNG"/>

## TOOLS

### Flask App for Saved Image and Live Camera Detection
To run the flask application:
```
python app.py
```
Open a web browser and type the below address:
```
http://127.0.0.1:80
```
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/flask_app_index.PNG"></img><br><br>
Upload files to save in a folder<br><br>
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/flask_app_upload.PNG"></img><br><br>
Show Detections<br><br>
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/flask_app_detections.PNG"></img><br><br>
<img src="https://github.com/DiversityDatabaseProject/Tensorflow_ObjectDetection/blob/main/img/flask_app_detections2.PNG"></img><br><br>
Click on "Home", and "Live Camera Detections" to see the camera for face detection.
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
python s3_utils.py --opt upload --from local/path --to bucket/folder
```
Note: Below script will download all files from S3 Bucket folder to the local folder.
```
python s3_utils.py --opt download --from bucket/folder --to local/path
```

## Resources
<a href="https://github.com/nicknochnack/TFODCourse">Full Object Detection Course Github Repo</a><br>
<a href="https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API">Tensorflow Lite Object Detection API Github Repo</a><br>
<a href="https://www.tensorflow.org/lite/convert">TFLite Conversion Article</a><br>
<a href="https://www.tensorflow.org/lite/examples/object_detection/overview"> TFLite Object Detection Article</a><br>
<a href="https://www.tensorflow.org/lite/performance/post_training_quantization">TFLite Post Training Quantization</a><br>
<a href="https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask">File Uploader</a><br>
<a href="https://github.com/smoh/imageviewer">Image Viewer</a><br>
<a href="https://github.com/tzutalin/labelImg">Image Labeller</a>

## Improvements needed

- include default values to python parameters, if none given by user
- test the bash scripts
- add error handling in the scripts
- add unit tests in python codes
