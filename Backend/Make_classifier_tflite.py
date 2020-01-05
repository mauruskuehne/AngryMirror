from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy import (array, dot, arccos, clip)
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import savez_compressed, load

emb_array = None
try:
   #emb_array = load("embeddings.npz")['arr_0']
   print(emb_array)
   print("loaded successfully")
    
except Exception:
    print("error")
    pass




datadir = '/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/output/'
dataset = facenet.get_dataset(datadir)
paths, labels = facenet.get_image_paths_and_labels(dataset)
print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))

if emb_array is  None:
    print('Loading feature extraction model')

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="/Users/samuelochsner/facerecognition/faceRecognition-yolo-facenet/model_mobile/angrymirror_big_facenet.tflite")
    interpreter.allocate_tensors()


    """ converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    with open(modelpath, 'wb') as modelfile:
        modelfile.write(tflite_quant_model) """

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    images_placeholder = input_details[0]['shape']

    embeddings = interpreter.get_tensor(output_details[0]['index'])

    embedding_size = embeddings.shape[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    batch_size = 1000
    image_size = 160
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        images = np.array(images, dtype=np.float32)
        for i in range(0, images.shape[0]):
            print("image {0}", i)
            imgs = images[i, :, :,:]
            imgs = np.expand_dims(imgs, axis=0)
            #imgs = tf.convert_to_tensor(imgs, np.int8) 
            try:              
                interpreter.set_tensor(input_details[0]['index'], imgs)
                interpreter.invoke()
                emb_array[i, :] = interpreter.get_tensor(output_details[0]['index'])
                print("processed {0}", i)
            except Exception as ex:
                print("Error with {0}", ex)
                pass




    savez_compressed('embeddings.npz', emb_array)

X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, test_size=0.5, random_state=42)

classifier_filename = './myclassifier/tiny_classifier.pkl'
classifier_filename_exp = os.path.expanduser(classifier_filename)

persons = [None, None, None, None]

for idx, label in enumerate(y_train):
    if label is 1:
        #jwo
        persons[1] = X_train[idx]
    if label is 0:
        #angeula
        persons[0] = X_train[idx]
    if label is 2:
        #maurus
        persons[2] = X_train[idx]
    if label is 3:
        #samuel
        persons[3] = X_train[idx]

""" 
for idx, data in enumerate(X_test):
    dist = -1
    index = 0
    for i, pers in enumerate(persons):
        d = np.linalg.norm(data-pers)
        angle = np.arccos(np.dot(data, pers) / (np.linalg.norm(data) * np.linalg.norm(pers)))

        print(d)
        if d > angle:
            dist = dot
            index = i

    print('Predicted {0} , actual {1}'.format(index, y_test[idx])) """
 

# Train classifier
print('Training classifier')
print(labels)
model = SVC(kernel='linear', probability=True)

model.fit(emb_array, labels)

score = model.score(X_test, y_test)
print(score)
# Create a list of class names
class_names = [cls.name.replace('_', ' ') for cls in dataset]

# Saving classifier model
with open(classifier_filename_exp, 'wb') as outfile:
    pickle.dump((model, class_names), outfile)
    pass
print('Saved classifier model to file "%s"' % classifier_filename_exp)
print('Goodluck')