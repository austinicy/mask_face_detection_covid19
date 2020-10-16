import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine

from threading import Timer
in_encoder = Normalizer('l2')
import tensorflow as tf
print("Using gpu: {0}".format(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)))
# initial values
CONFIDENCE=0.5
THRESHOLD=0.3

# set parameter
probability_minimum = 0.5
threshold = 0.3
min_dist = 1

class MaskDetector:

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

    #get face embedding and perform face recognition
    def get_embedding(image,model_Face):
        # scale pixel values
        face = image.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = cv2.resize(face,(160,160))
        face = np.expand_dims(face, axis=0)
        encode = model_Face.predict(face)[0]
        return encode

    def find_person(encoding,database):
        min_dist = float("inf")
        encoding = in_encoder.transform(np.expand_dims(encoding, axis=0))[0]
        for (name, db_enc) in database.items():
            dist = cosine(db_enc,encoding)
            if dist < 0.5 and dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.5:
            return "None"
        else:
            return identity
        return "None"

    def detect(self, frame, net, ln, LABELS, COLORS, W, H, model_Face, faceCascade, database):
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        #faces_list = []
        #encodes = []
        names = []
                
        # def hello():
        #     print("Hello")
        # t = Timer(5.0, hello) 
        # t.start()
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    if classID == 0 :
                        classIDs.append(classID)
                        names.append('')
                    elif classID == 1:
                        #openCV
                        #convert to greyscale
                        #faces_list=[]
                        #encodes=[]
                        crop = frame[y:y+int(height), x:x+int(width)]

                        #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        #detect face
                        #faces = faceCascade.detectMultiScale(crop,
                                                            #scaleFactor=1.1,
                                                            #minNeighbors=5,
                                                            ##minSize=(60, 60),
                                                            #flags=cv2.CASCADE_SCALE_IMAGE)
                        #to draw rectangle
                        label = ""
                        #for (x, y, w, h) in faces:
                        #face_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_frame = img_to_array(crop)
                        name = "None"
                        if face_frame.size!=0 :
                            face_frame = cv2.resize(face_frame,(160, 160))
                            encode = MaskDetector.get_embedding(face_frame,model_Face)
                            name = MaskDetector.find_person(encode,database)
                        if name == "None":
                            label = "Not found"
                        else :
                            label = name

                        classIDs.append(classID)
                        names.append(label)
                        print(label)

        # apply non-maximal suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]]+":"+names[i], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 10)