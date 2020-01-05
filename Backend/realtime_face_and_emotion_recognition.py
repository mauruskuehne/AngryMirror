from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from scipy import misc
import cv2 as cv
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import time
import imageio
import PIL
from PIL import Image
from PIL import *
from utils import *
import zmq

#zeromq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
frame_interval = 3
batch_size = 1000
image_size = 182
input_image_size = 160

print('Loading feature extraction model')
modelpath = "/home/pi/OpenVino/test/model_mobile/angrymirror_big_facenet.tflite"


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=modelpath)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data.
images_placeholder = input_details[0]['shape']

embeddings = interpreter.get_tensor(output_details[0]['index'])

embedding_size = embeddings.shape[1]

classifier_filename = '/home/pi/OpenVino/test/myclassifier/tiny_classifier.pkl'
classifier_filename_exp = os.path.expanduser(classifier_filename)
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)
    print('load classifier file-> %s' % classifier_filename_exp)



# Load the model.
net = cv.dnn.readNet('/home/pi/OpenVino/test/models/face-detection-adas-0001.xml',
                     '/home/pi/OpenVino/test/models/face-detection-adas-0001.bin')

emotion_net = cv.dnn.readNet('/home/pi/OpenVino/test/models/emotions-recognition-retail-0003_1.xml', 
                            '/home/pi/OpenVino/test/models/emotions-recognition-retail-0003_1.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
emotion_net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

#Fonts
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 128, 255)
lineType = 2

# Read an image.
cap = cv.VideoCapture(0)

def send_prediction(predicted_person, predicted_emotion):
    try:
        socket.send_json(
                        {
                            "name": predicted_person,
                            "emotion": predicted_emotion
                        }
                        )
    except Exception as ex:
        print(ex)
        pass


while(True):
    for i in range(30):
        cap.grab()
    ret, frame = cap.read()

    #frame = cv.imread('/path/to/image')
    if frame is None:
        raise Exception('Image not found!')
    # Prepare input blob and perform an inference.
    blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
    net.setInput(blob)
    out = net.forward()
    # Draw detected faces on the frame.
    detections = out.reshape(-1, 7)

    confidentials = np.array(detections[:,2])
    top_confidentials = confidentials[confidentials > 0.5].argsort()

    if len(top_confidentials) == 0:
        send_prediction('nobody', 'neutral')
        continue

    y_max = np.array(detections[:,5])
    y_min = np.array(detections[:,3])
    
    v = y_max - y_min
    n = np.argmax(v[top_confidentials])
    # print(v[top_confidentials])
    # print(n)
    # print(np.max(v))
    # print(confidentials[n])
    # y = detections[:,4] - y
    detection = detections[n]
    # print(detection[6] - detection[4])
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])


    if xmin < 1 or xmax < 1 or ymin < 1 or ymax < 1 or confidence < 0.5:
        continue
    print("y" + str((ymax - ymin)))
    print("x" + str((xmax - xmin)))
    if (xmax - xmin) < 25:
        print("person found is to small (x axis)")
        continue
    
    face = frame[ymin:ymax, xmin:xmax]

    face_blob = cv.dnn.blobFromImage(face, size=(64, 64), ddepth=cv.CV_8U)
    emotion_net.setInput(face_blob)
    out = emotion_net.forward()

    emotion = np.argmax(out)
    

    face_detect = cv.resize(face,(160, 160))
    emb_array = np.zeros((1, embedding_size))

    #scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
    #scaled = cv2.resize(scaled, (input_image_size,input_image_size),
    #                    interpolation=cv2.INTER_CUBIC)
    scaled = facenet.prewhiten(face_detect)
    scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))

    scaled_reshape = np.array(scaled_reshape, dtype=np.float32)
    for i in range(0, scaled_reshape.shape[0]):

        imgs = scaled_reshape[i, :, :,:]
        imgs = np.expand_dims(imgs, axis=0)

        try:              
            interpreter.set_tensor(input_details[0]['index'], imgs)
            interpreter.invoke()
            emb_array[i, :] = interpreter.get_tensor(output_details[0]['index'])
            print("processed ", i)
        except Exception:
            print("Error with ", i)
            pass

    predictions = model.predict_proba(emb_array)
    if np.max(predictions, axis=1) < 0.7:
        print("Not sure who is in front of the camera")
        continue
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    #print(best_class_probabilities)
    predicted_person = class_names[best_class_indices[0]]

    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
    
    bottomLeftCornerOfText = (xmin, ymin)

    predicted_emotion = emotions[emotion]

    cv.putText(frame, predicted_person + " " + predicted_emotion, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    print('Person predicted: ' + predicted_person)
    print('Emotion predicted: ' + predicted_emotion)
    send_prediction(predicted_person, predicted_emotion)

    #cv.imshow('frame',frame)
    #if cv.waitKey(1) & 0xFF == ord('q'):
    #    break