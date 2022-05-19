#!/bin/bash
# This batch file will prepare and call the training script
# To run, type .\win_scripts\train_configs.ps1 on windows powershell from root dir of repository
echo "===================================="
echo "Create Training directories"
echo "===================================="
CFG_FILE=utils/config.ini
CFG_CONTENT=$(cat $CFG_FILE | sed -r '/[^=]+=[^=]+/!d' | sed -r 's/\s+=\s/=/g')
eval "$CFG_CONTENT"
mkdir -p "$TRAIN_TEST_IMGS"
mkdir -p "$TRAIN_TRAIN_IMGS"
mkdir -p "$TFLITE_INFERENCE_RES"
echo "Create directories - DONE."
echo "====================================="
echo "Clone git repo generate record script"
echo "====================================="
# Prepare training data
pushd -Path "$TF_ROOT" 
#removes weird error message
git clone "$TRAIN_TFREC_REPO"
popd
echo "Cloning git - DONE."
echo "===================================="
echo "Prepare training data"
echo "===================================="
# Create the train data
python $TRAIN_TFREC_PATH/generate_tfrecord.py -x $TRAIN_TRAIN_IMGS -l $TRAIN_ANNOTATIONS/label_map.pbtxt -o $TRAIN_ANNOTATIONS/train.record
# Create the test data
python $TRAIN_TFREC_PATH/generate_tfrecord.py -x $TRAIN_TEST_IMGS -l $TRAIN_ANNOTATIONS/label_map.pbtxt -o $TRAIN_ANNOTATIONS/test.record
# Run the model_configs script
python model_configs.py
# Copy Model Config to Training Folder
cp $MODEL_FPN_PATH/pipeline.config $CUSTOM_MODEL_PATH
cp $PROTOC_PATH/readme.txt $TEST_PATH