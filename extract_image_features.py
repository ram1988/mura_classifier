from skimage import transform, io
import csv
import os, pickle
import tensorflow as tf
import numpy as np
from cnn_classifier import *


# Feature extractor
def extract_features(image_path, vector_size=75):
    image = io.imread(image_path, as_gray=True)
    image = transform.resize(image,(vector_size,vector_size),mode='symmetric',preserve_range=True)
    return image

def indices_to_one_hot(data, nb_classes=2):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

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

def serving_input_rvr_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[100], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec ={"images": tf.FixedLenFeature([75,75], tf.float32)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


'''
train_dataset = prepare_image_set("MURA-v1.1/train_labeled_studies.csv","train_dataset")
validation_dataset = prepare_image_set("MURA-v1.1/valid_labeled_studies.csv","validation_dataset")
'''
print("extract.......--->")
train_image_features = []
train_image_labels = []
tot = 0
for i in range(1,9):
    train_records = pickle.load(open("train_dataset"+str(i)+".pkl", 'rb'))
    for record in train_records:
        train_image_features.append(np.array(record[0]))
        train_image_labels.append(int(record[1]))


train_image_labels = indices_to_one_hot(train_image_labels)
print(train_image_labels[0])

#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
vector_size=75
num_classes = 2
cnnclassifier = CNNClassifier(vector_size,num_classes)
model = cnnclassifier.train_model()

train_image_features = train_image_features[0:100]
train_image_labels = train_image_labels[0:100]

print(train_image_features[0].shape)
print(len(train_image_features))
print(len(train_image_labels))

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': np.array(train_image_features)}, y=np.array(train_image_labels),
    batch_size=100, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=2000)
model.export_savedmodel("test_model",serving_input_receiver_fn=serving_input_rvr_fn)


'''
# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)



'''