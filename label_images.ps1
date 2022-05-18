# This batch file will download and install the scripts (from git repository) for image labeling
# To run, type .\label_images.ps1 on windows powershell
# Make sure you are in the git repository's root directory
Write-Host "====================================" -ForegroundColor Green
Write-Host "Creating directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
mkdir -p Tensorflow/workspace/images/collectedimages/face
mkdir -p Tensorflow/labelImg
Write-Host "====================================" -ForegroundColor Green
Write-Host "Installing python dependencies" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
pip install --upgrade pyqt5 lxml
Write-Host "====================================" -ForegroundColor Green
Write-Host "Cloning lblImg github repo" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
Push-Location -Path "Tensorflow"
git clone https://github.com/tzutalin/labelImg
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
Write-Host "Finished with the setup." -ForegroundColor Green
Write-Host "Launching the ImgLabel app..." -ForegroundColor Green
python labelImg.py