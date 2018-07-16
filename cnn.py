# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:15:55 2018

@author: rutwi
"""

#import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Part 1 - Building the Convolutional Neural Network

#Importing Keras modules and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3 , input_shape = (64, 64, 3), activation = "relu"))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3 , activation = "relu"))
#Adding second Max Polling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = "relu"))

classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Part2 - Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataSET/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                               'dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')



# checkpoint
#from keras.callbacks import ModelCheckpoint
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


classifier.fit_generator(
                            training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set ,
                            validation_steps=2000
                            )
