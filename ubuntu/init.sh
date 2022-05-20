#!/bin/bash
# This batch file will install the python package requirements, create the Tensorflow workspace directory, and download files to build Tensorflow and the required libraries
# To run, type .\ubuntu\init.sh on the terminal from root dir of repository
# Section 1: Install required packages
echo "===================================="
echo "Installing requirements.txt contents"
echo "===================================="
pip3 install --upgrade pip
pip3 install -r requirements.txt
echo "Installing requirements.txt - DONE."
echo "========================================="
echo "Creating Tensorflow workspace directories"
echo "========================================="
# format and read config file
CFG_FILE=utils/config.ini
CFG_CONTENT=$(cat $CFG_FILE | sed -r '/[^=]+=[^=]+/!d' | sed -r 's/\s+=\s/=/g')
eval "$CFG_CONTENT"
mkdir -p $PROTOC_PATH
mkdir -p $PRE_TRAINED_MODELS_PATH
echo "Tensorflow workspace directories - DONE."
# Section 3: Download Tensorflow models
echo "===================================="
echo "Download Tensorflow models"
echo "===================================="
pushd Tensorflow 
#removes weird error message
git clone $TF_MODELS_REPO
popd
echo "Downloading tensorflow models - DONE."
# Section 4: Download and install protocol buffers
echo "===================================="
echo "Download and install protocol buffers"
echo "===================================="
sudo apt-get update
sudo apt-get install protobuf-compiler
pushd $TF_MODELS_RESEARCH
protoc object_detection/protos/*.proto --python_out=.
echo "Download and install protocol buffers - DONE."
# Section 5: Build Tensorflow libraries
echo "===================================="
echo "Build Tensorflow libraries"
echo "===================================="
cp object_detection/packages/tf2/setup.py setup.py
python setup.py build
python setup.py install
cd slim
pip3 install -e .
popd
echo "Build Tensorflow libraries - DONE."
echo "Initialization finished."
