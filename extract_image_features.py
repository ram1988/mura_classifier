from skimage import transform, io
import csv
import os, pickle
import tensorflow as tf
import numpy as np
from cnn_classifier import *

vector_size=200
num_classes = 2
batch_size = 100

# Feature extractor
def extract_features(image_path, vector_size=200):
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

        pickle.dump(image_train_labels, open(file_name + str(batch_num) + ".pkl", 'wb'))

    return image_train_labels

def serving_input_rvr_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[batch_size], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec ={"images": tf.FixedLenFeature([vector_size,vector_size], tf.float32)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

#train_dataset = prepare_image_set("MURA-v1.1/train_labeled_studies.csv","train_dataset")
#validation_dataset = prepare_image_set("MURA-v1.1/valid_labeled_studies.csv","validation_dataset")
print("extract.......--->")
train_image_features_glb = []
val_image_features_glb = []
train_image_labels_glb = []
val_image_labels_glb = []
tot = 0
for i in range(1,9):
    train_records = pickle.load(open("train_dataset"+str(i)+".pkl", 'rb'))
    for record in train_records:
        train_image_features_glb.append(np.array(record[0]))
        train_image_labels_glb.append(int(record[1]))

for i in range(1,3):
    val_records = pickle.load(open("validation_dataset"+str(i)+".pkl", 'rb'))
    for record in val_records:
        val_image_features_glb.append(np.array(record[0]))
        val_image_labels_glb.append(int(record[1]))

train_image_labels_glb = indices_to_one_hot(train_image_labels_glb)
val_image_labels_glb = indices_to_one_hot(val_image_labels_glb)
print(train_image_labels_glb[0])

#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
cnnclassifier = CNNClassifier(vector_size,num_classes)
model = cnnclassifier.get_classifier_model()

#train_size = int(len(train_image_features)*0.8)
train_size = 100
print(train_size)



#https://medium.com/@vincentteyssier/tensorflow-estimator-tutorial-on-real-life-data-aa0fca773bb
#change the logic accordingly
def train_input_fn(features, labels, batch_size, repeat_count):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(256).repeat(repeat_count).batch(batch_size)
    return dataset

# input_fn for evaluation and predicitions (labels can be null)
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

def train():
    train_image_features = train_image_features_glb[0:100]
    train_image_labels = train_image_labels_glb[0:100]
    train_image_labels = np.array(train_image_labels)
    train_image_labels = np.reshape(train_image_labels,(-1,2))
    print(train_image_labels.shape)
    print("shapes")
    print(len(train_image_features))
    print(len(train_image_labels))

    steps = (len(train_image_features)/batch_size)-1
    steps = steps if steps>0  else 1
    print(steps)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': np.array(train_image_features)}, y=train_image_labels,
        batch_size=100, num_epochs=10, shuffle=False)
    # Train the Model
    model.train(input_fn,steps = steps)
    print(model)

def evaluate():
    val_image_features = val_image_features_glb
    print("evaluate...")
    print(len(val_image_features))
    val_image_labels = val_image_labels_glb
    val_image_labels = np.array(val_image_labels)
    val_image_labels = np.reshape(val_image_labels, (-1, 2))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': np.array(val_image_features)}, y=val_image_labels,
        batch_size=50,shuffle=False)

    steps = (len(val_image_features) / batch_size) - 1
    steps = steps if steps > 0  else 1

    # Train the Model
    evaluate_result = model.evaluate(input_fn, steps=steps)
    print ("Evaluation results")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))
    #model.export_savedmodel("test_model",serving_input_receiver_fn=serving_input_rvr_fn)


def predict():
   pred_image_features = train_image_features_glb[train_size:][0:3]
   #pred_image_features = np.array(pred_image_features)
   print("3")
   print(len(pred_image_features))
   print(pred_image_features[0].shape)
   model_input = tf.train.Example(features=tf.train.Features(feature={"images":tf.train.Feature(float_list=tf.train.FloatList(value=pred_image_features[0]))}))
   model_input = model_input.SerializeToString()
   predictor = tf.contrib.predictor.from_saved_model("test_model/1545839240")
   output_dict = predictor({"predictor_inputs": [model_input]})
   print(output_dict)


train()
evaluate()
#predict()
