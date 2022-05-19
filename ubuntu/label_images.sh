#!/bin/bash
# This batch file will download and install the scripts (from git repository) for image labeling
# To run, type .\win_scripts\label_images.ps1 on windows powershell from root dir of repository
echo "===================================="
echo "Creating directories"
echo "===================================="
# format and read config file
CFG_FILE=utils/config.ini
CFG_CONTENT=$(cat $CFG_FILE | sed -r '/[^=]+=[^=]+/!d' | sed -r 's/\s+=\s/=/g')
eval "$CFG_CONTENT"
mkdir -p $LABEL_FACE_IMGS
mkdir -p $LABEL_IMGS_PATH
echo "===================================="
echo "Installing python dependencies"
echo "===================================="
pip install --upgrade pyqt5 lxml
echo "===================================="
echo "Cloning lblImg github repo"
echo "===================================="
#removes weird error message
pushd $TF_ROOT
git clone $LABEL_IMG_URL
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
echo "Finished with the setup."
echo "Launching the ImgLabel app..."
python labelImg.py