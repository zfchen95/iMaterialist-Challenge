
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

# filename class_id.jpg
def random_generator(rootpath, batch_size):
    if rootpath[-1] != '/':
        rootpath = rootpath + '/'
    imagesname = os.listdir(rootpath)
    imagesname.remove('.DS_Store')
    
    while True: # generator only called once, not every epoch. So need an infinite loop
        y_data = np.zeros((batch_size,1), dtype=np.int)
        x_data = np.zeros((batch_size,224,224,3), dtype=np.float32)

        samples = random.sample(imagesname, batch_size)
        print(samples)
        for idx, sample in enumerate(samples):
            y_data[idx] = int(sample.split("_")[0])
            x_data[idx] = readImage(rootpath+sample)
        
        y_data_binary = keras.utils.np_utils.to_categorical(y_data, num_classes = 128)

        yield (x_data, y_data_binary)


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
    nb_epoch = 1800
    steps = 100 #steps*nb_epoch = 200000 > 19000, can cover the whole dataset
    result=my_model.fit_generator(random_generator(rootpath, batch_size),                                                      
                        steps_per_epoch=steps,
                        epochs = nb_epoch, 
                        validation_data=random_generator(rootpath, batch_size),
                        validation_steps = 2,
                        )

if __name__ == '__main__':
    main()