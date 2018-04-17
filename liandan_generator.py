
# coding: utf-8

# In[1]:


import cv2
import csv
import keras 
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
import numpy as np
from keras.models import Model
from sklearn.model_selection import train_test_split
import random


# In[2]:


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


# In[77]:


# load data
# returns: a list of data [x_data, y_data]
def random_generator(self, image_path, anno_path, batch_size):

    # load annotation into memory first
    annotations = {}
    with open(anno_path, 'r+') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            annotations[row[0]] = int(row[1])-1 #-1 here to make correct binary catogory

    size = len(annotations)
    print("annotation size", size)
    
    while True: # generator only called once, not every epoch. So need an infinite loop
        y_data = np.zeros((batch_size,1), dtype=np.int)
        x_data = np.zeros((batch_size,224,224,3), dtype=np.float32)

        count = 0
        rand_keys = random.sample(list(annotations), batch_size * 10) # here, *10 is to avoid invalid img_url
        for key in rand_keys:
            # stop get a new data, since we have get a full batch
            if count >= batch_size:
                break
            im = readImage(image_path+key+'.jpg')
            if im is None:
                continue
            im = np.expand_dims(im, axis=0)
            x_data[count] = im

            label = annotations[key]
            label_np = np.array([label])
            label_np = np.expand_dims(label_np, axis=0)
            y_data[count] = label_np

            count += 1

        if count < batch_size:
            print("not enough valid img_url")

        y_data_binary = keras.utils.np_utils.to_categorical(y_data, num_classes = 128)

        yield (x_data, y_data_binary)
#     return [x_data, y_data_binary]

def main():
    training_path = sys.argv[1]
    training_annotation = sys.argv[2]
    validate_path = sys.argv[3]
    validate_annotation = sys.argv[4]

    # create the base pre-trained model
    dense_model = keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False)
    # dense_model.summary()

    # add a global spatial average pooling layer
    x = dense_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 128 classes
    predictions = Dense(128, activation='softmax')(x)

    # this is the model we will train
    my_model = Model(inputs=dense_model.input, outputs=predictions)
    #my_model.summary()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in dense_model.layers:
        layer.trainable = False
    #my_model.summary() # trainable params decrease

    my_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


    # train the model on the new data for a few epochs
    batch_size = 32
    nb_epoch = 2000
    steps = 100 #steps*nb_epoch = 200000 > 19000, can cover the whole dataset
    result=my_model.fit_generator(self.random_generator(training_path, training_annotation, batch_size),                                                      
                            steps_per_epoch=steps,
                            epoch = nb_epoch, 
                            validation_data=self.random_generator(validate_path, validate_annotation, batch_size),
                            validation_steps = 2
                            )

if __name__ == '__main__':
    main()