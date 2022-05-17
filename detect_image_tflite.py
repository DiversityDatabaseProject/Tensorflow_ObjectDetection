######## Webcam Object Detection Using Tensorflow-trained Classifier #########
# Source: https://github.com/JerryKurata/TFlite-object-detection/blob/main/TFLite_detection_image.py
# Author: Evan Juras
# Date: 9/28/19
# Description: 
# This program uses a TensorFlow Lite object detection model to perform object 
# detection on an image or a folder full of images. It draws boxes and scores 
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# 
# Edited by Maria Rosario Sebastian
# 05/06/2022

# Import packages
import os
import cv2
import numpy as np
import glob
import tensorflow as tf


TEST_IMAGE_PATH = os.path.join('Tensorflow', 'workspace','images', 'detect_image')
MODEL_NAME = 'detect_cl.tflite'
MODEL_PATH = os.path.join('Tensorflow', 'workspace','models','my_ssd_mobnet', 'tfliteexport',MODEL_NAME)
LABELMAP_NAME = 'labels.txt'
LABEL_PATH = os.path.join('Tensorflow', 'workspace','models','my_ssd_mobnet', 'tfliteexport',LABELMAP_NAME)
DETECT_RES = os.path.join('Tensorflow', 'workspace','images','detect_tflite_res')
min_conf_threshold = float(.5)

# Get path to current working directory
CWD_PATH = os.getcwd()

# Parse input image name and directory. 
IM_NAME = 'istockphoto_174878359.jpg'
IM_DIR = os.path.join(CWD_PATH,TEST_IMAGE_PATH)

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABEL_PATH)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Loop over every image and perform detection
for image_path in images:    
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects, 0
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects, 5
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects, 1
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    #print('*************boxes: ', boxes)
    #print('*************classes: ', classes)
    #print('*************scores: ', scores)
    #print('*************len(scores): ', len(scores))
    #print('*************labels[int(classes[0])]: ',labels[int(classes[0])])
    
    ctr=0
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # All the results have been drawn on the image, now display the image
    #cv2.imshow('Object detector', image)
    filename=image_path.split('\\')[-1]
    image_name = os.path.join(CWD_PATH,DETECT_RES,filename)
    cv2.imwrite(image_name, image)

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
#cv2.destroyAllWindows()