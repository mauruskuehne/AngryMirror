from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import time
import zmq


from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/yolo_cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/align/model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
args = parser.parse_args()

print('Creating networks and loading parameters')
# Load YOLO V2 model.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#zeromq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
frame_interval = 3
batch_size = 1000
image_size = 182
input_image_size = 160

print('Loading feature extraction model')
modelpath = "/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/model_mobile/angrymirror_big_facenet.tflite"


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

classifier_filename = '/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/myclassifier/tiny_classifier.pkl'
classifier_filename_exp = os.path.expanduser(classifier_filename)
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)
    print('load classifier file-> %s' % classifier_filename_exp)

#video_capture = cv2.VideoCapture('/Users/samuelochsner/yoloface/outputs/_yoloface.avi')
video_capture = cv2.VideoCapture(0)

c = 0

print('Start Recognition!')
prevTime = 0
while True:
    ret, frame = video_capture.read()

    # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

    curTime = time.time()    # calc fps
    timeF = frame_interval

    if (c % timeF == 0):
        find_results = []

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]

        # Use YOLO to get bounding boxes
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        bounding_boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        nrof_faces = len(bounding_boxes)

        if nrof_faces > 0:
            img_size = np.asarray(frame.shape)[0:2]

            bb = np.zeros((nrof_faces,4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = bounding_boxes[i][0]
                bb[i][1] = bounding_boxes[i][1]
                bb[i][2] = bounding_boxes[i][2]
                bb[i][3] = bounding_boxes[i][3]

                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('face is inner of range!')
                    continue

                cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                cropped = facenet.flip(cropped, False)
                scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
                scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))

                scaled_reshape = np.array(scaled_reshape, dtype=np.float32)
                print(scaled_reshape.shape)
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

                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print(best_class_probabilities)
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                text_x = bb[i][0]
                text_y = bb[i][3] + 20


                result_names = class_names[best_class_indices[0]]
                try:
                    socket.send(result_names.encode())
                except Exception as ex:
                    print(ex)
                    pass
                
                #print(result_names)
                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 255), thickness=1, lineType=2)
        else:
            try:              
                socket.send(b"nobody")
            except Exception as ex:
                print(ex)
                pass

            print('Unable to align')

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, str, (text_fps_x, text_fps_y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
    # c+=1
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
# #video writer
# out.release()
cv2.destroyAllWindows()
