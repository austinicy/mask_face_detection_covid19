import os
import threading
import argparse

from flask import Flask, Response
from flask import flash, request, redirect, jsonify
from flask import render_template

from werkzeug.utils import secure_filename

from models.realStream import RealStream
from models.facenet import FaceNet

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

TEMPLATES_AUTO_RELOAD = True

# initialize a flask object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/realStream/")
def realStream():
    # start a thread that will perform mask detection
    t = threading.Thread(target=RealStream.mask_detection)
    t.daemon = True
    t.start()
    # forward to real stream page
    return render_template("realStream.html")


@app.route("/staticStream/")
def staticStream():
    # forward to static stream page
    return render_template("staticStream.html")

@app.route("/about/")
def about():
    # forward to about page
    return render_template("about.html")

@app.route("/contact/")
def contact():
    # forward to contact page
    return render_template("contact.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(RealStream.generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/content_dash", methods=['GET'])
def content_dash():
    data = request.values
    if data['type'] == 'imagecode':
        return render_template('imagecode.html')
    if data['type'] == 'imageprocess':
        return render_template('imageprocess.html')
    if data['type'] == 'folderscan':
        return render_template('folderscan.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'uploadFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['uploadFile']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # encoding
            md = FaceNet()
            username = request.form['username']
            md.save_encode_db(username, filename)
        return jsonify('success')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# execute function
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ip address")
    ap.add_argument("-o", "--port", type=int, default=8000, help="port number of the server")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)