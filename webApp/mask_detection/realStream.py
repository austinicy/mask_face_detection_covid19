import os
import time
import cv2
import csv
import threading
import datetime
import imutils
import numpy as np
import tensorflow as tf;
from imutils.video import FPS
from mask_detection.webcamVideoStream import WebcamVideoStream
from tensorflow.keras.models import load_model
from mask_detection.detector import MaskDetector

# setup the path for YOLOv4
YOLO_PATH="yolov4"
OUTPUT_FILE="output/outfile.avi"

# load the class labels our YOLO model was trained
labelsPath = os.path.sep.join([YOLO_PATH, "/data/classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_PATH, "/yolov4_custom_train_best.weights"])
configPath = os.path.sep.join([YOLO_PATH, "/cfg/yolov4_custom_test.cfg"])

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


#Facenet
#load facenet pre-train database
FACENET_PATH="facenet"
dbPath = os.path.sep.join([FACENET_PATH, "dict.csv"])
reader = csv.reader(open(dbPath), delimiter='\n')
database = {}
for row in reader:
    data = row[0].split(",", 1)
    encode = data[1].replace('"','')
    encode = encode.replace('[','')
    encode = encode.replace(']','')
    encode = np.fromstring(encode, dtype=float, sep=',')
    database[data[0]] = encode

facePath = os.path.sep.join([FACENET_PATH, "facenet_keras.h5"])
#model2 = load_model("/Users/austin/Desktop/NUS/Project/mask_recog_ver2.h5",custom_objects={ 'loss': triplet_loss })
#to read the classifier
cvPath = os.path.sep.join([FACENET_PATH, "haarcascade_frontalface_alt2.xml"])
faceCascade = cv2.CascadeClassifier(cvPath)

# start = time.time()
model_Face = load_model(facePath, custom_objects={ 'loss': MaskDetector.triplet_loss })
# end = time.time()
# print("loading model complete, time: ".format(end-start))




class RealStream:
    # read frame from video
    def mask_detection():
        # global references to the video stream, output frame, and lock variables
        global vs, outputFrame, lock

        # initialize the video stream and allow the camera sensor to warmup
        vs = WebcamVideoStream(src=0).start()
        fps = FPS().start()
        time.sleep(2.0)

        # initialize the detection and the total number of frames read thus far
        (W, H) = (None, None)
        md = MaskDetector()

        # loop over frames from the video stream
        th = threading.currentThread()
        while getattr(th, "running", True):
            # read the next frame from the video stream
            frame = vs.read()

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # grab the current timestamp and draw it on the frame
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


            
            input_frame = frame
            output_frame = frame
            greyed = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            greyed = np.dstack([greyed, greyed, greyed])

            # GaussianBlur  -- not helpful
            # greyed = cv2.GaussianBlur(greyed, (21, 21), 0)

            #  sharpen
            # http://datahacker.rs/004-how-to-smooth-and-sharpen-an-image-in-opencv/
            filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(greyed,-1,filter)

            md.detect(sharpened,output_frame, net, ln, LABELS, COLORS, W, H, model_Face, faceCascade, database)

            # resize the frame, for output
            output_frame = imutils.resize(output_frame, width=400)


            # cv2.putText(frame, "HELLO",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
            # acquire the lock, set the output frame, and release the lock
            with lock:
                outputFrame = output_frame.copy()

            if frame is None:
                break
        print("thread is stopped, stopping camera")
        vs.stop()

    # plot the frame onto video
    def generate():
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

# release the video stream pointer
#vs.stop() 
