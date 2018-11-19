from skimage import transform, io
import csv
import os, pickle
import tensorflow as tf
import numpy as np
from cnn_classifier import *


# Feature extractor
def extract_features(image_path, vector_size=200):
    image = io.imread(image_path, as_gray=True)
    image = transform.resize(image,(vector_size,vector_size),mode='symmetric',preserve_range=True)
    return image

def prepare_image_set(path,file_name):

    with open(path) as train_labels:
        csv_reader = csv.reader(train_labels)
        i = 0
        batch_num = 1
        image_train_labels = []
        for row in csv_reader:
            image_files = os.listdir(row[0])
            label = row[1]
            for image in image_files:
                image_file = row[0]+image
                image_features = extract_features(image_file)
                image_train_labels.append((image_features,label))
                i = i+1
                print(i)
                if len(image_train_labels) == 5000:
                    print("written")
                    pickle.dump(image_train_labels, open(file_name+str(batch_num)+".pkl", 'wb'))
                    batch_num = batch_num+1
                    image_train_labels = []

        pickle.dump(image_train_labels, open(file_name + str(batch_num+1) + ".pkl", 'wb'))

    return image_train_labels


#train_dataset = prepare_image_set("MURA-v1.1/train_labeled_studies.csv","train_dataset")
#validation_dataset = prepare_image_set("MURA-v1.1/valid_labeled_studies.csv","validation_dataset")

train_image_features = []
train_image_labels = []
tot = 0
for i in range(1,9):
    train_records = pickle.load(open("train_dataset"+str(i)+".pkl", 'rb'))
    for record in train_records:
        train_image_features.append(np.array(record[0]))
        train_image_labels.append(record[1])


#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
vector_size=200
num_classes = 2
cnnclassifier = CNNClassifier(vector_size,num_classes)
model = cnnclassifier.train_model()

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': np.array(train_image_features)}, y=np.array(train_image_labels),
    batch_size=100, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=2000)

'''
# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)



'''