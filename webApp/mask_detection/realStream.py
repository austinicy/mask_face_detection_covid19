import os
import time
import cv2
import csv
import threading
import datetime
import imutils
import numpy as np
import tensorflow as tf;

from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from mask_detection.models import MaskDetector

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
weightsPath = os.path.sep.join([YOLO_PATH, "/yolov4_custom_train_final.weights"])
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


class RealStream:
    #Load model
    def triplet_loss(y_true, y_pred, alpha = 0.2):

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss

    def mask_detection():
        model_Face = load_model(facePath, custom_objects={ 'loss': RealStream.triplet_loss })

        # global references to the video stream, output frame, and lock variables
        global vs, outputFrame, lock

        # initialize the video stream and allow the camera sensor to warmup
        #vs = VideoStream(usePiCamera=1).start()
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # initialize the detection and the total number of frames read thus far
        (W, H) = (None, None)
        md = MaskDetector()

        # loop over frames from the video stream
        while True:
            # read the next frame from the video stream
            frame = vs.read()

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # grab the current timestamp and draw it on the frame
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # call function to detect the mask of frames read thus far
            #md.detect(frame, net, ln, LABELS, COLORS, W, H)
            md.detect(frame, net, ln, LABELS, COLORS, W, H, model_Face, faceCascade, database)

            # resize the frame
            frame = imutils.resize(frame, width=400)

            # acquire the lock, set the output frame, and release the lock
            with lock:
                outputFrame = frame.copy()

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
