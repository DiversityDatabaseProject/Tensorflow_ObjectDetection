from werkzeug.utils import secure_filename
import imghdr
import os
from flask import Flask,Response , request, render_template, redirect, send_from_directory
from flask import Flask
import detect_from_cam as det
import load_configs as cf
import detect_images_metadata as metadata
from pager import Pager
import detect_from_image as detect
import csv, datetime

app = Flask(__name__, template_folder='./templates')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = cf.IMAGE_UPLOAD
#app.config['IMAGE_UPLOAD_RES'] = cf.IMAGE_UPLOAD_RES

#create the folders for the image detection results
date = datetime.datetime.now()
print(date.timestamp())
datetime_string = str(date).split(' ')[0]+str(date.timestamp()).split('.')[0]+'_'+str(date.timestamp()).split('.')[1]
resfoldername='detection_res_'+datetime_string
uploadfoldername='saved_images_'+datetime_string
#saved uploaded images path
IMAGES_UPLOAD = os.path.join('static', 'images', uploadfoldername)
os.makedirs(IMAGES_UPLOAD)
#name and path of image detection results
DETECTION_RESULTS_PATH=os.path.join('static', 'images', resfoldername)
app.config['IMAGE_UPLOAD_RES'] = resfoldername
os.makedirs(DETECTION_RESULTS_PATH)
csvname = 'image_metadata_'+datetime_string+'.csv' #Name of saved csv
#name and path of csv file
TABLE_FILE = os.path.join('static',csvname)
print('DETECTION_RESULTS_PATH: ', DETECTION_RESULTS_PATH)
print('TABLE_FILE: ', TABLE_FILE)

@app.route('/goto', methods=['POST', 'GET'])    
def goto():
    return redirect('/' + request.form['index'])

def validate_image(stream):
    '''
    validates file uploaded
    '''
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def read_table(url):
    """Return a list of dict"""
    # r = requests.get(url)
    with open(url) as f:
        return [row for row in csv.DictReader(f.readlines())]

@app.errorhandler(413)
def too_large(e):
    '''Checks uploaded image size'''
    return "File is too large", 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stop_cam')
def stop_cam():
    return render_template('index.html')

@app.route('/face_detection_image')
def face_detection_image():
    '''Upload images for face detection'''
    files = os.listdir(DETECTION_RESULTS_PATH)
    return render_template('file_upload.html', files=files)

@app.route('/face_detection_image', methods=['POST'])
def upload_files():
    '''saves and displays uploaded images for face detection'''
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(IMAGES_UPLOAD, filename))
    return '', 204

@app.route('/show_detections', methods=['GET'])
def show_detections():
    '''Performs face detection on images in folder'''
    checkpoint_file=os.path.join(cf.paths['CHECKPOINT_PATH'], 'ckpt-51')
    detect.face_detection(checkpoint=checkpoint_file, labelmap=cf.files['LABELMAP'], test_images=IMAGES_UPLOAD, detect_res=DETECTION_RESULTS_PATH, min_thresold=float(.5))
    #call the metadata creation function
    metadata.create_image_metadata(resultspath=DETECTION_RESULTS_PATH, csvpath=TABLE_FILE)
    return redirect('/0')

@app.route('/page_viewer')
def page_viewer():
    return redirect('/0')

@app.route('/<int:ind>/')
def image_view(ind=None):
    table = read_table(TABLE_FILE)
    pager = Pager(len(table))
    if ind >= pager.count:
        return render_template("404.html"), 404
    else:
        pager.current = ind
        return render_template(
            'image_viewer.html',
            index=ind,
            pager=pager,
            data=table[ind], images_folder=app.config['IMAGE_UPLOAD_RES'])

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(IMAGES_UPLOAD, filename)

@app.route('/detect_cam')
def detect_cam():
    app.logger.debug('Running camera detection')
    det.detect()
    return redirect('/')

if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)