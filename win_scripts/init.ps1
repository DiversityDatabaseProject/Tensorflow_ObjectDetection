# This batch file will install the python package requirements, create the Tensorflow workspace directory, and download files to build Tensorflow and the required libraries
# To run, type .\win_scripts\init.ps1 on windows powershell from root dir of repository
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
Push-Location utils
Foreach ($i in $(Get-Content config.ini)){
    Set-Variable -Name $i.split("=")[0] -Value $i.split("=",2)[1]
}
Pop-Location
mkdir -p $PROTOC_PATH
mkdir -p $PRE_TRAINED_MODELS_PATH
Write-Host "Tensorflow workspace directories - DONE." -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Create Training directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
mkdir -p $TRAIN_TEST_IMGS
mkdir -p $TRAIN_TRAIN_IMGS
mkdir -p $TFLITE_INFERENCE_RES
Write-Host "Create directories - DONE." -ForegroundColor Green
# Section 4: Download Tensorflow models
Write-Host "====================================" -ForegroundColor Green
Write-Host "Download Tensorflow models" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Push-Location -Path "Tensorflow" 
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
git clone $TF_MODELS_REPO
Pop-Location
Write-Host "Downloading tensorflow models - DONE." -ForegroundColor Green
# Section 5: Download and install protocol buffers
Write-Host "====================================" -ForegroundColor Green
Write-Host "Download and install protocol buffers" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Invoke-WebRequest $PROTOC_BUFFERS_URL -OutFile $PROTOC_BUFFERS_FILE
# Extract protocol buffers zip file
Expand-Archive -Path $PROTOC_BUFFERS_FILE -DestinationPath $PROTOC_PATH
# Add protoc bin folder to environment path
$env:Path = (Get-Location).path+'\Tensorflow\protoc\bin;' + $env:Path
Push-Location -Path $TF_MODELS_RESEARCH
protoc object_detection/protos/*.proto --python_out=.
Write-Host "Download and install protocol buffers - DONE." -ForegroundColor Green
# Section 6: Build Tensorflow libraries
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