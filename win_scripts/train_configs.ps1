# This batch file will prepare and call the training script
# To run, type .\train_configs.ps1 on windows powershell
# Make sure you are in the git repository's root directory
Write-Host "====================================" -ForegroundColor Green
Write-Host "Create Training directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
mkdir -p Tensorflow/workspace/images/test
mkdir -p Tensorflow/workspace/images/train
mkdir -p Tensorflow/workspace/images/detect_tflite_res
Write-Host "Create directories - DONE." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Clone git repo generate record script" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
# Prepare training data
Push-Location -Path "Tensorflow" 
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
git clone https://github.com/nicknochnack/GenerateTFRecord
Remove-Item -Path .\GenerateTFRecord\.git -Force -Recurse -Confirm:$false
Pop-Location
Write-Host "Cloning git - DONE." -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Prepare training data" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
# Create the train data
python Tensorflow/GenerateTFRecord/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
# Create the test data
Create the test data
python Tensorflow/GenerateTFRecord/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
# Run the model_configs script
python model_configs.py
# Copy Model Config to Training Folder
cp Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config Tensorflow/workspace/models/my_ssd_mobnet
