# This batch file will prepare and call the training script
# To run, type .\win_scripts\train_configs.ps1 on windows powershell from root dir of repository
Write-Host "====================================" -ForegroundColor Green
Write-Host "Create Training directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Push-Location utils
Foreach ($i in $(Get-Content config.ini)){
    Set-Variable -Name $i.split("=")[0] -Value $i.split("=",2)[1]
}
Pop-Location
mkdir -p $TRAIN_TEST_IMGS
mkdir -p $TRAIN_TRAIN_IMGS
mkdir -p $TFLITE_INFERENCE_RES
Write-Host "Create directories - DONE." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Clone git repo generate record script" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
# Prepare training data
Push-Location -Path $TF_ROOT 
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
git clone $TRAIN_TFREC_REPO
Remove-Item -Path .\GenerateTFRecord\.git -Force -Recurse -Confirm:$false
Pop-Location
Write-Host "Cloning git - DONE." -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host "Prepare training data" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
# Create the train data
python $TRAIN_TFREC_PATH/generate_tfrecord.py -x $TRAIN_TRAIN_IMGS -l $TRAIN_ANNOTATIONS/label_map.pbtxt -o $TRAIN_ANNOTATIONS/train.record
# Create the test data
Create the test data
python $TRAIN_TFREC_PATH/generate_tfrecord.py -x $TRAIN_TEST_IMGS -l $TRAIN_ANNOTATIONS/label_map.pbtxt -o $TRAIN_ANNOTATIONS/test.record
# Run the model_configs script
python model_configs.py
# Copy Model Config to Training Folder
cp $MODEL_FPN_PATH/pipeline.config $CUSTOM_MODEL_PATH
cp $PROTOC_PATH/readme.txt $TEST_PATH