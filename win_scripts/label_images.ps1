# This batch file will download and install the scripts (from git repository) for image labeling
# To run, type .\win_scripts\label_images.ps1 on windows powershell from root dir of repository
Write-Host "====================================" -ForegroundColor Green
Write-Host "Creating directories" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Push-Location utils
Foreach ($i in $(Get-Content config.ini)){
    Set-Variable -Name $i.split("=")[0] -Value $i.split("=",2)[1]
}
Pop-Location
mkdir -p $LABEL_FACE_IMGS
mkdir -p $LABEL_IMGS_PATH
Write-Host "====================================" -ForegroundColor Green
Write-Host "Installing python dependencies" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
pip install --upgrade pyqt5 lxml
Write-Host "====================================" -ForegroundColor Green
Write-Host "Cloning lblImg github repo" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
#removes weird error message
$env:GIT_REDIRECT_STDERR = '2>&1'
Push-Location -Path $TF_ROOT
git clone $LABEL_IMG_URL
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
Write-Host "Finished with the setup." -ForegroundColor Green
Write-Host "Launching the ImgLabel app..." -ForegroundColor Green
python labelImg.py