import os
import time
import cv2
import threading
import datetime
import imutils
import numpy as np

from imutils.video import VideoStream,FPS
from models.webcamVideoStream import WebcamVideoStream
from models.facenet import FaceNet
from models.detector import MaskDetector
from models.util import utils

from flask import Response
# setup the path for YOLOv4

# load the class labels our YOLO model was trained
labelsPath = os.path.sep.join(["webApp/cfg", "classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["webApp/data", "yolov4.weights"])
configPath = os.path.sep.join(["webApp/cfg", "yolov4.cfg"])

# load our YOLO object detector and determine only the *output* layer names
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
vs = None
outputFrame = None
lock = threading.Lock()

class RealStream:
    def __init__(self):
        self.maskDetector = MaskDetector()

    def init_config(self):
        None

    def mask_detection(self):
        # global references to the video stream, output frame, and lock variables
        global vs, outputFrame, lock

        # initialize the video stream and allow the camera sensor to warmup
        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        time.sleep(2.0)

        # initialize the detection and the total number of frames read thus far
        (W, H) = (None, None)

        # loop over frames from the video stream
        th = threading.currentThread()
        while getattr(th, "running", True):
            # read the next frame from the video stream
            frame = vs.read()
            
            frame = self.processFrame(frame, W, H)

            # resize the frame, for output
            frame = imutils.resize(frame, width=400)


            # cv2.putText(frame, "HELLO",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
            # acquire the lock, set the output frame, and release the lock
            with lock:
                outputFrame = frame.copy()

            if frame is None:
                break
        print("thread is stopped, stopping camera")
        vs.stop()

    # plot the frame onto video
    def generate(self):
        # grab global references to the output frame and lock variables
        global outputFrame, lock

        # loop over frames from the output stream
        while True:
            # wait until the lock is acquired
            with lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if outputFrame is None:
                    continue

                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
        


    # process frame
    def processFrame(self, frame, W, H):
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # grab the current timestamp and draw it on the frame
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


            # greyed = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            # greyed = np.dstack([greyed, greyed, greyed])

            # GaussianBlur  -- not helpful
            # greyed = cv2.GaussianBlur(greyed, (21, 21), 0)

            #  sharpen
            # http://datahacker.rs/004-how-to-smooth-and-sharpen-an-image-in-opencv/
            # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # sharpened = cv2.filter2D(greyed,-1,filter)

            # call function to detect the mask of frames read thus far
            self.maskDetector.detect(frame, net, ln, LABELS, COLORS, W, H)

            return frame

    # process uploaded image / video
    def processimage(self, filename):
        print("process image for -> " + filename)

        # read image
        filepath = utils.get_file_path('webApp/uploads', filename)
        print(filepath)
        image = cv2.imread(filepath)

        # process frame
        frame = self.processFrame(image, None, None)

        # generate processed image
        basename = os.path.splitext(filename)[0]
        outputfile = basename+"_processed.jpg"

        cv2.imwrite(utils.get_file_path('webApp/uploads', outputfile), frame)
        print("processed image was successfully saved")

        return outputfile

    # process uploaded image / video
    def processvideo(self, filename):
        print("process video for -> " + filename)

        # read video file
        filepath = utils.get_file_path('uploads', filename)

        # generate processed file name
        outputfilename = os.path.splitext(filename)[0] + "_processed.mp4"
        outputfilepath = utils.get_file_path('uploads', outputfilename)

        # read from video file
        video = cv2.VideoCapture(filepath)
        fps = FPS().start()

        # initial parameters
        writer = None
        (H, W) = (None, None)

        while True:
            (grabbed, frame) = video.read()

            if not grabbed:
                break

            # resize frame to width=300
            frame = imutils.resize(frame, width=300)

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # check whether writer is None
            if writer is None:
                writer = cv2.VideoWriter(
                                    filename=outputfilepath,
                                    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps=video.get(cv2.CAP_PROP_FPS),
                                    frameSize=(W, H))

            # process the frame and update the FPS counter
            frame = self.processFrame(frame, W, H)

            cv2.imshow("frame", frame)

            writer.write(frame)

            cv2.waitKey(1)
            fps.update()

        # do a bit of cleanup
        fps.stop()
        cv2.destroyAllWindows()
        writer.release()

        print("processed video was successfully saved")

        return outputfilename

# release the video stream pointer
#vs.stop() 
