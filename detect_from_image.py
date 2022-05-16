import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from common_functions import detect_fn, detection_model

#cluster model folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

LABEL_MAP_NAME = 'label_map.pbtxt'

DETECTED_IMAGE_NAME = 'detection_test.png'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'), # high level workspace
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'), # where TF record file will be stored
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'), # folder for the selected models tested
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'DETECT_RES_PATH': os.path.join('Tensorflow', 'workspace','images','detect_res')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'DETECTED_IMAGE': os.path.join(paths['DETECT_RES_PATH'], DETECTED_IMAGE_NAME)
}

# Load pipeline config and build a detection model
#configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
#detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# Selecting our most train model
#ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-2')).expect_partial()
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-51')).expect_partial()

# select our face label
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
#print(category_index)

# Reset image path
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'detect_image', 'istockphoto_174878359.jpg')
#print(IMAGE_PATH)

# Loading image into python
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

# Converting image to a tensor and the detection function 
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

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
            min_score_thresh=.2,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.savefig(files['DETECTED_IMAGE'])
#plt.show()