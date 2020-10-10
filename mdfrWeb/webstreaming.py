import threading
import argparse

from flask import Response
from flask import Flask
from flask import render_template
from mask_detection.realStream import RealStream

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

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(RealStream.generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


# execute function
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="port number of the server")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)