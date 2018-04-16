
# coding: utf-8

import cv2
import csv
import sys
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
import numpy as np
from keras.models import Model
from keras import backend as K


# use GPU option
# K.tensorflow_backend._get_available_gpus()


# read in and preprocess image
def readImage(pathname):
    oriim = cv2.imread(pathname)
    if oriim is None:
        return None
    im = cv2.resize(oriim, (224, 224)).astype(np.float32)
    # Subtract mean pixel and multiple by scaling constant
    # Reference: https://github.com/shicai/DenseNet-Caffe
    im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
    im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
    im[:,:,2] = (im[:,:,2] - 123.68) * 0.017
    return im



# load data
# input: option 'train' or 'validate
# returns: a list of data [x_data, y_data]
def loadData(image_path, anno_path):

    # load annotation into memory first
    annotations = {}
    with open(anno_path, 'r+') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            annotations[row[0]] = int(row[1])-1 #-1 here to make correct binary catogory

    size = len(annotations)
    print("annotation size", size)

    y_data = np.zeros((size,1), dtype=np.int)
    x_data = np.zeros((1500,224,224,3), dtype=np.float32)

    count = 0
    for key,value in annotations.items():
        im = readImage(image_path+key+'.jpg')
        # handle error
        if im is None:
            continue
        im = np.expand_dims(im, axis=0)
        x_data[count] = im

        label = value
        label_np = np.array([label])
        label_np = np.expand_dims(label_np, axis=0)
        y_data[count] = label_np

        count += 1

    count -= 1
    print("valid image count:",count)

    x_data = x_data[:count]
    y_data = y_data[:count]
    print('x_data shape:', x_data.shape)
    print('y_data shape:', y_data.shape)

    return [x_data,y_data]


def main():
    training_path = sys.argv[1]
    training_annotation = sys.argv[2]
    validate_path = sys.argv[3]
    validate_annotation = sys.argv[4]

    # load training set and validation set
    [x_train,y_train] = loadData(training_path, training_annotation)
    [x_val, y_val] = loadData(validate_path, validate_annotation)


    # create the base pre-trained model
    dense_model = keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False)
    #dense_model.summary()

    # add a global spatial average pooling layer
    x = dense_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a logistic layer -- let's say we have 128 classes
    predictions = Dense(128, activation='softmax')(x)

    # this is the model we will train
    my_model = Model(inputs=dense_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in dense_model.layers:
        layer.trainable = False
    #my_model.summary() # trainable params decrease

    my_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    y_train_binary = keras.utils.np_utils.to_categorical(y_train)
    y_val_binary = keras.utils.np_utils.to_categorical(y_val)

    print y_train.shape
    print y_val.shape
    print y_train_binary.shape
    print y_val_binary.shape

    # train the model on the new data for a few epochs
    batch_size = 32
    nb_epoch = 200
    result=my_model.fit(x_train, y_train_binary,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(x_val, y_val_binary),
                      shuffle=True)

if __name__ == '__main__':
    main()
