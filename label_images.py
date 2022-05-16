# Import opencv
import cv2 
import uuid
import os
import time

#creating new path where will store our image to label
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages', 'face')

# create the file path 
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

#We will be using labelImg tool to annotate our image, it requires the following package 
#lxml : when label images we create an xml file containing the label data (original image size, depth, object name, number of label per image, bonding box)
#!pip3 install --upgrade pyqt5 lxml

#creating labelimg path
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')


