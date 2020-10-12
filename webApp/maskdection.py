import threading
import argparse

from flask import Flask, Response
from flask import flash, request, redirect
from flask import render_template
from models.realStream import RealStream

# initialize a flask object
app = Flask(__name__)

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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        else:
            flash("pending.....")


# execute function
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ip address")
    ap.add_argument("-o", "--port", type=int, default=8000, help="port number of the server")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)