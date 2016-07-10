import json
import os

caffe_root = 'tps/crfasrnn/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')


from os import listdir
from os.path import isfile, join
import shutil
import tarfile
import tempfile

from flask import Flask, flash, request, redirect, render_template, session, abort, url_for, send_file
from hashlib import sha256
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from subprocess import call
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

from PIL import Image as PILImage

from models import User

#pipeline
import lzma
import cv2
from tps.crfasrnn.utils import semantic_segmentation

UPLOAD_FOLDER = 'files/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
NUM_SCALE = 4.0
NUM_SCALE_DOWN = 1.0 / NUM_SCALE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
engine = create_engine('sqlite:///users.db', echo=True)


@app.route('/', methods=['GET'])
def index():
    if not session.get('logged_in'):
        return render_template('index.html')
    else:
        return redirect(url_for('files'))

@app.route('/login', methods=['GET'])
def login():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return redirect(url_for('index'))

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/do_register', methods=['POST'])
def do_register():
    username = str(request.form['username'])
    password = str(request.form['password'])

    # create a Session
    Session = sessionmaker(bind=engine)
    session = Session()
     
    user = User(username, password)
    session.add(user)

    return redirect(url_for('login'))
 

@app.route('/do_login', methods=['POST'])
def do_login():
    username = str(request.form['username'])
    password = str(request.form['password'])
 
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).filter(User.username.in_([username]), User.password.in_([password]) )
    result = query.first()
    if result:
        session['logged_in'] = True
    else:
        flash('Unrecognized account! Please try again.')
    return redirect(url_for('index'))

@app.route("/do_logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/files', methods=['GET'])
def files():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        file_names = set([f for f in listdir(UPLOAD_FOLDER) if isfile(join(UPLOAD_FOLDER, f))])
        file_names -= set(['.no_content'])
        for f in file_names:
            f = f[:7]
        file_info = {}
        for file in file_names:
            filesize = os.stat(UPLOAD_FOLDER + '/' + file).st_size / float(10**6)
            file_info[file] = filesize

        return render_template('files.html', data=file_info)

@app.route('/upload', methods=['GET'])
def upload():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('upload.html')

@app.route('/do_upload', methods=['POST'])
def do_upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath_original = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        filepath_downsampled = os.path.join(app.config['UPLOAD_FOLDER'], 'downsampled-' + secure_filename(file.filename))
        filepath_upsampled = os.path.join(app.config['UPLOAD_FOLDER'], 'upsampled-' + secure_filename(file.filename))
        filepath_maskimage = os.path.join(app.config['UPLOAD_FOLDER'], 'maskimage-' + secure_filename(file.filename))
        filepath_tarfile = os.path.join(app.config['UPLOAD_FOLDER'], 'tar-' + secure_filename(file.filename) + '.tar')
        filepath_lzma = filepath_tarfile + '.xz'

        file.save(filepath_original)
        src = cv2.imread(filepath_original)

        print type(src)
        dest_inter_cubic = cv2.resize(src, None, fx=NUM_SCALE_DOWN, fy=NUM_SCALE_DOWN, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(filepath_original.replace(secure_filename(file.filename), "downsampled-"+secure_filename(file.filename)), dest_inter_cubic)

        src = cv2.imread(filepath_downsampled)
        dest_inter_cubic = cv2.resize(src, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(filepath_upsampled, dest_inter_cubic)

        upsampled_image = PILImage.open(filepath_upsampled)
        if  upsampled_image.size[0] < 500:
            mask_image = semantic_segmentation.find_subjects(filepath_original)
            plt.imshow(mask_image)
            plt.show()
            w, h = mask_image.size
            for x in range(0, w):
                for y in range(0, h):
                    pixel = mask_image.getpixel((x,y))
                    if pixel[0] > 0 or pixel[1] > 0 or pixel[2] > 0:
                        upscalePixel = upsampled_image.getpixel((x,y))
                        r = pixel[0]# - upscalePixel[0]
                        g = pixel[1]# - upscalePixel[1]
                        b = pixel[2]# - upscalePixel[2]
                        mask_image.putpixel((x,y), (r,g,b))
            mask_image.save(filepath_maskimage)
        else:
            mask_image = PILImage.new("RGB", upsampled_image.size, "black")
            mask_image.save(filepath_maskimage)

        tar = tarfile.open(filepath_tarfile, "w")
        for name in [filepath_maskimage, filepath_downsampled]:
            tar.add(name)
        tar.close()

        # compress with lzma
        call(["xz", filepath_tarfile])

        #os.remove(filepath_original)
        #os.remove(filepath_downsampled)
        #os.remove(filepath_upsampled)
       # os.remove(filepath_maskimage)
        return redirect(url_for('files'))

@app.route('/do_download', methods=['GET'])
def download():

    temp = "files/uploaded/tmp.jpg"
    filename_original = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('name'))

    print request.args.get('name')

    call(["tar", "xf", filename_original])
    filepath_maskimage = os.path.join(app.config['UPLOAD_FOLDER'], 'maskimage-' +  request.args.get('name')[4:-7])
    filepath_downsampled = os.path.join(app.config['UPLOAD_FOLDER'],'downsampled-' +  request.args.get('name')[4:-7])

    src = cv2.imread(filepath_downsampled)
    dest_inter_cubic = cv2.resize(src, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(temp, dest_inter_cubic)

    mask_image = PILImage.open(filepath_maskimage)
    mask_image = mask_image.convert('RGB')
    plt.imshow(mask_image)
    plt.show()
    temp_image = PILImage.open(temp)
    temp_image = temp_image.convert('RGB')
    plt.imshow(temp_image)
    plt.show()

    w, h = mask_image.size
    if w < 500:
        for x in range(0, w):
            for y in range(0, h):
                pixel = mask_image.getpixel((x,y))
                if pixel != (0,0,0):
                    r = pixel[0]# + temp_pixel[0]
                    g = pixel[1]# + temp_pixel[1]
                    b = pixel[2]# + temp_pixel[2]
                    temp_image.putpixel((x,y), (r,g,b))

    plt.imshow(temp_image)
    plt.show()

    os.remove(filepath_maskimage)
    os.remove(filepath_downsampled)

    return send_file(temp, attachment_filename=filename_original[4:-7], as_attachment=True)

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8080, debug=True)
