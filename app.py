import json
import os
from os import listdir
from os.path import isfile, join
import tempfile

import cv2
from flask import Flask, flash, request, redirect, render_template, session, abort, url_for, send_file
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from werkzeug.utils import secure_filename

from models import User


UPLOAD_FOLDER = 'files/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
NUM_SCALE = 4
NUM_SCALE_DOWN = 1 / NUM_SCALE

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
        filename = secure_filename(file.filename)

        src = cv2.imread(filename)
        dest_inter_cubic = cv2.resize(src, None, fx=NUM_SCALE_DOWN, fy=NUM_SCALE_DOWN)
        cv2.imwrite(filename, dest_inter_cubic)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('files'))

@app.route('/do_download', methods=['GET'])
def download():
    filename = request.args.get('name')
    file = (UPLOAD_FOLDER + '/' + filename)

    expanded_fp = tempfile.TemporaryFile()
    src = cv2.imread(file)
    dest_inter_cubic = cv2.resize(src, None, fx=NUM_SCALE, fy=NUM_SCALE, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(expanded_fp.name, dest_inter_cubic)

    return send_file(expanded_fp.name, attachment_filename=filename, as_attachment=True)

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=5001, debug=True)