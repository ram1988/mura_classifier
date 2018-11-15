from skimage import transform, io
import csv
import os, pickle

#from cnn_classifier import *

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


tot = 0
for i in range(1,9):
    obj = pickle.load(open("train_dataset"+str(i)+".pkl", 'rb'))
    tot = tot+len(obj)
print(tot)
obj = pickle.load(open("validation_dataset2.pkl", 'rb'))
print(len(obj))

print(len(train_dataset))
print(len(validation_dataset))



'''
vector_size=200
num_classes = 7
cnnclassifier = CNNClassifier(vector_size,3,num_classes)
cnnclassifier.create_features()
cnnclassifier.train_classifier()
'''
