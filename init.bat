@ECHO OFF 
:: This batch file will install the python package requirements, create the Tensorflow workspace directory, and download files to build Tensorflow and the required libraries
TITLE Installing requirements.txt
:: Section 1: Install required packages
ECHO ==========================
ECHO Installing requirements.txt contents
ECHO ============================
ECHO
ECHO Please wait... installing requirements.txt for python package dependencies...
ECHO
pip3 install --upgrade pip
pip3 install -r requirements.txt
ECHO Installing requirements.txt - DONE.
TITLE Create Tensorflow workspace and build directories
:: Section 1: Create Tensorflow workspace and build directories
ECHO ==========================
ECHO Installing requirements.txt contents
ECHO ============================
ECHO
ECHO Creating Tensorflow workspace directories...
mkdir -p Tensorflow/protoc
mkdir -p Tensorflow/workspace/pre-trained-models
mkdir -p Tensorflow/scripts
ECHO Tensorflow workspace directories - DONE.
ECHO Downloading tensorflow models...
git clone https://github.com/tensorflow/models Tensorflow
ECHO Downloading tensorflow models - DONE.
ECHO Download and install protocol buffers...
Invoke-WebRequest https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protoc-3.20.1-win64.zip -OutFile Tensorflow/protoc/protoc-3.20.1-win64.zip
:: Extract protocol buffers zip file
Expand-Archive -Path Tensorflow/protoc/protoc-3.20.1-win64.zip -DestinationPath Tensorflow/protoc/
:: Add protoc bin folder to environment path
$env:Path = (Get-Location).path+'\Tensorflow\protoc\bin;' + $env:Path
Push-Location -Path "Tensorflow/models/research" 
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py setup.py
python setup.py build
python setup.py install
cd slim
pip3 install -e .
Pop-Location
