'''
This script detects all images in a specified folder and outputs them in another folder
Example:
python detect_from_image.py  --checkpoint Tensorflow\workspace\models\my_ssd_mobnet\ckpt-51 --label_map Tensorflow\workspace\annotations\label_map.pbtxt --threshold .5 --images_folder Tensorflow\workspace\images\detect_image --output_path Tensorflow\workspace\images\detect_res
Source: https://github.com/nicknochnack/TFODCourse
Modifications: Maria Rosario SEBASTIAN, May 2022
- added parameter inputs (argparse)
- detecting multiple images from a given folder, instead of just one
- saves the images in a given folder instead of showing them as a pop-up window
'''
import os
import argparse
import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import load_configs as cf
import glob
import tensorflow as tf
from object_detection.builders import model_builder

def detect_fn(image, detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def face_detection(checkpoint, labelmap, test_images, detect_res, min_thresold):
    # Create detection results folder
    if not os.path.exists(detect_res):
        os.makedirs(detect_res)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(cf.files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # Selecting our most train model
    ckpt.restore(checkpoint).expect_partial()

    # select our face label
    category_index = label_map_util.create_category_index_from_labelmap(labelmap)

    # Define path to images and grab all image filenames
    images = glob.glob(test_images + '/*')

    # Loop over every image and perform detection
    for image_path in images: 
        # Loading image into python
        img = cv2.imread(image_path)
        #print('img: ', img)
        image_np = np.array(img)

        # Converting image to a tensor and the detection function 
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=8,
                    min_score_thresh=min_thresold,
                    agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        image_path.replace("\\","/")
        filename=image_path.split('/')[-1]
        image_name = os.path.join(detect_res,filename)
        plt.savefig(image_name)

def main(args):
    '''
    python detect_from_image.py  
    --checkpoint Tensorflow\workspace\models\my_ssd_mobnet\ckpt-51 
    --label_map Tensorflow\workspace\annotations\label_map.pbtxt 
    --threshold .5 
    --images_folder Tensorflow\workspace\images\detect_image 
    --output_path Tensorflow\workspace\images\detect_res
    '''
    
    CHECKPOINT = args['checkpoint']
    LABEL_MAP = args['label_map']
    TEST_IMAGE_PATH = args['images_folder']
    DETECT_RES_PATH = args['output_path']
    PARAM_THRESHOLD = args['threshold']

    MIN_THRESHOLD = float(.5)
    if PARAM_THRESHOLD is not None:
        MIN_THRESHOLD = float(PARAM_THRESHOLD)
    
    face_detection(checkpoint=CHECKPOINT, labelmap=LABEL_MAP, test_images=TEST_IMAGE_PATH, detect_res=DETECT_RES_PATH, min_thresold=MIN_THRESHOLD)


if __name__ == '__main__':
            # create parser and handle arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', required=True, help='path\\to\\checkpoint\\checkpoint_file, without the file extension, eg: path\ckpt-51')
        parser.add_argument('--label_map', required=True, help='path\\to\\label_map.pbtxt')
        parser.add_argument('--threshold', required=False, help='detection score threshold, eg: .5')
        parser.add_argument('--images_folder', required=True, help='path\\to\\image\\folder')
        parser.add_argument('--output_path', required=True, help='path\\to\\inference\results\\folder')

        args = vars(parser.parse_args())
        
        main(args)