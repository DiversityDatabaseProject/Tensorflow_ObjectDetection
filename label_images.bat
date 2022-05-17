@ECHO OFF 
:: This batch file will download and install the scripts (from git repository) for image labeling
TITLE Installing requirements.txt
ECHO ============================
ECHO Creating directories
ECHO ============================
mkdir -p Tensorflow/workspace/images/collectedimages/face
mkdir -p Tensorflow/labelImg
pip install --upgrade pyqt5 lxml
ECHO ============================
ECHO Cloning lblImg github repo
ECHO ============================
git clone https://github.com/tzutalin/labelImg Tensorflow
Push-Location -Path "Tensorflow/labelImg"
pyrcc5 -o libs/resources.py resources.qrc
ECHO Done!
ECHO Launching the app...
python labelImg.py
