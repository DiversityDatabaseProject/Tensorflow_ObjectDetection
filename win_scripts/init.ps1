# This batch file will install the python package requirements, create the Tensorflow workspace directory, and download files to build Tensorflow and the required libraries
# To run, type .\init.ps1 on windows powershell
# Make sure you are in the git repository's root directory
# Section 1: Install required packages
Write-Host "====================================" -ForegroundColor Green
Write-Host "Installing requirements.txt contents" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
pip3 install --upgrade pip
pip3 install -r requirements.txt
Write-Host "Installing requirements.txt - DONE." -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Creating Tensorflow workspace directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
mkdir -p Tensorflow/protoc
mkdir -p Tensorflow/workspace/pre-trained-models
Write-Host "Tensorflow workspace directories - DONE." -ForegroundColor Green
# Section 3: Download Tensorflow models
Write-Host "====================================" -ForegroundColor Green
Write-Host "Download Tensorflow models" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Push-Location -Path "Tensorflow" 
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
git clone https://github.com/tensorflow/models
Pop-Location
Write-Host "Downloading tensorflow models - DONE." -ForegroundColor Green
# Section 4: Download and install protocol buffers
Write-Host "====================================" -ForegroundColor Green
Write-Host "Download and install protocol buffers" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Invoke-WebRequest https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protoc-3.20.1-win64.zip -OutFile Tensorflow/protoc/protoc-3.20.1-win64.zip
# Extract protocol buffers zip file
Expand-Archive -Path Tensorflow/protoc/protoc-3.20.1-win64.zip -DestinationPath Tensorflow/protoc/
# Add protoc bin folder to environment path
$env:Path = (Get-Location).path+'\Tensorflow\protoc\bin;' + $env:Path
Push-Location -Path "Tensorflow/models/research" 
protoc object_detection/protos/*.proto --python_out=.
Write-Host "Download and install protocol buffers - DONE." -ForegroundColor Green
# Section 5: Build Tensorflow libraries
Write-Host "====================================" -ForegroundColor Green
Write-Host "Build Tensorflow libraries" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
cp object_detection/packages/tf2/setup.py setup.py
python setup.py build
python setup.py install
cd slim
pip3 install -e .
Pop-Location
Write-Host "Build Tensorflow libraries - DONE." -ForegroundColor Green
Write-Host "Initialization finished." -ForegroundColor Green