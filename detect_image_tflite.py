'''
Webcam Object Detection Using Tensorflow-trained Classifier #########
Source: https://github.com/JerryKurata/TFlite-object-detection/blob/main/TFLite_detection_image.py
Author: Evan Juras
Date: 9/28/19
Description: 
This program uses a TensorFlow Lite object detection model to perform object 
detection on an image or a folder full of images. It draws boxes and scores 
around the objects of interest in each image.

This code is based off the TensorFlow Lite image classification example at:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
I added my own method of drawing boxes and labels using OpenCV.

Edited by Maria Rosario Sebastian
05/06/2022
- added input parameters (argparse)
- corrected the indexes of the output for scores, bbox coordinates, class and num detections
- saves detected images in a given folder instead of showing them on pop-up window. 
''' 


import os
import argparse
import cv2
import numpy as np
import glob
import tensorflow as tf

def main(args):
    TFLITE_MODEL = args['tf_model']
    TF_LABELMAP = args['tf_labels']
    IM_DIR = args['images_folder']
    TF_INFERENCE_RES_FOLDER = args['output_path']
    PARAM_THRESHOLD = args['threshold']
    MIN_CONF_THRESHOLD = float(.5)
    if PARAM_THRESHOLD is not None:
        MIN_CONF_THRESHOLD = float(PARAM_THRESHOLD)

    # Define path to images and grab all image filenames
    if IM_DIR:
        PATH_TO_IMAGES = IM_DIR
        images = glob.glob(PATH_TO_IMAGES + '/*')
        
    if not os.path.exists(TF_INFERENCE_RES_FOLDER):
        os.makedirs(TF_INFERENCE_RES_FOLDER)

    # Load the label map
    with open(TF_LABELMAP, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)
    #print('floating_model: ',floating_model)
    #print('input_details: ',input_details)
    #print('output_details: ',output_details)
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
        #print('input_data: ',input_data)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        #print('invoke interpreter: ')
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects, 0
        print('*************boxes: ', boxes)
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects, 5
        print('*************classes: ', classes)
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects, 1
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        #print('*************scores: ', scores)
        #print('*************len(scores): ', len(scores))
        #print('*************labels[int(classes[0])]: ',labels[int(classes[0])])
        
        ctr=0
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0)):
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

        # get the filename and save the image with detections to folder
        filename=image_path.replace("\\","/")
        image_name = os.path.join(TF_INFERENCE_RES_FOLDER,filename.split('/')[-1])
        print(image_name)
        cv2.imwrite(image_name, image)

if __name__ == '__main__':
    # create parser and handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_model', required=True, help='path\\to\\models\\tflitemodelname.tflite')
    parser.add_argument('--tf_labels', required=True, help='path\\to\\labelfile\\labels.txt')
    parser.add_argument('--threshold', required=False, help='detection score threshold, eg: .5')
    parser.add_argument('--images_folder', required=True, help='path\\to\\image\\folder')
    parser.add_argument('--output_path', required=True, help='path\\to\\inference\results\\folder')

    args = vars(parser.parse_args())
    
    main(args)