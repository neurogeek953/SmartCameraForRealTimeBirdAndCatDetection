#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:49:52 2021

@author: TEB
"""

# numpy import Classic :P
import numpy as np


# Keras imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


## functions to easily set up network parameters

# First Convolutional layer parameters
def select_1st_convolutional_layer_args(output_neurons, window_size, input_neurons, activation_function = 'relu'):
    return [output_neurons, window_size, input_neurons, activation_function]

# Second Convolutional layer parameters
def select_2nd_convolutional_layer_args(output_neurons, window_size, activation_function = 'relu'):
    return [output_neurons, window_size, activation_function]

# 
def select_pooling_layer_args(poolX, poolY):
    return (poolX, poolY)

def select_hidden_layer_args(nb_neurons, activation_function = 'relu'):
    return [nb_neurons, activation_function]

class CNN_Classifier:
    def __init__(self,
                 first_convolutional_layer_arguments,
                 first_pooling_layer_arguments,
                 second_convolutional_layer_arguments,
                 second_pooling_layer_arguments,
                 hidden_layer_argument,
                 output_layer_argument):
        
        self.inputSize = (first_convolutional_layer_args[2][0], first_convolutional_layer_args[2][0])
        
        # Setup the classifier
        self.classifier = Sequential()
        
        # Layer 1: 1st Convolutional layer
        self.classifier.add(Conv2D(first_convolutional_layer_args[0],
                                   first_convolutional_layer_args[1],
                                   input_shape = first_convolutional_layer_args[2],
                                   activation = first_convolutional_layer_args[3]))
 
        # Layer 2: 1st Pooling Layer
        self.classifier.add(MaxPooling2D(pool_size = first_pooling_layer_args))
        
        # Layer 3: 2nd Convolutional Layer
        self.classifier.add(Conv2D(second_convolutional_layer_args[0],
                                   second_convolutional_layer_args[1],
                                   activation = second_convolutional_layer_args[2]))
        
        # Layer 4: 2nd Pooling Layer
        self.classifier.add(MaxPooling2D(pool_size = second_pooling_layer_args))
        
        # Layer 5: 1st Flattening Layer
        self.classifier.add(Flatten())
        
        # Layer 6: 1st Hidden Layer (Classic Artificial Neural Network)
        self.classifier.add(Dense(units = hidden_layer_argument[0],
                                  activation = hidden_layer_argument[1]))
        
        # Layer 7: Output Layer
        if output_layer_argument == 1:
            # make a one output network using a sigmoid activation function
            # If the result is 1 we have a bird otherwise it is a cat
            self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
            # condition setup for the predict method
            self.outputType = 1
        if output_layer_argument == 0:
            # make a two output node network, the one with the highest score is the cat/bird
            self.classifier.add(Dense(units = 2, activation = 'sigmoid'))
            # condition set up for the predict method
            self.outputType = 0
        
        # Compile the CNN
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    def make_training_and_test_sets(self,
                                    scaling = 1/255.,
                                    Range = 0.2,
                                    batch_size = 16,
                                    training_set_path = 'dataset/training',
                                    test_set_path = 'dataset/test'):
        
        # If the network is has one output node
        if self.outputType == 1:
            
            # Set up the training set generator.
            train_datagen = ImageDataGenerator(rescale = scaling,
                                               shear_range = Range,
                                               zoom_range = Range,
                                               horizontal_flip = True)
            # Set up the test set generator
            test_datagen = ImageDataGenerator(rescale = scaling)
            
            # Make the training set
            self.trainingSet = train_datagen.flow_from_directory(training_set_path,
                                                                 target_size = self.inputSize,
                                                                 batch_size = batch_size,
                                                                 class_mode = 'binary')
            
            # Make the test set
            self.testSet = test_datagen.flow_from_directory(test_set_path,
                                                            target_size = self.inputSize,
                                                            batch_size = batch_size,
                                                            class_mode = 'binary')
            
            
            
            
        # If the network is has two output nodes
        if self.outputType == 0:
            
            # Set up the training set generator.
            train_datagen = ImageDataGenerator(rescale = scaling,
                                               shear_range = Range,
                                               zoom_range = Range,
                                               horizontal_flip = True)
            
            # Set up the test set generator
            test_datagen = ImageDataGenerator(rescale = scaling)
            # Make the training set
            self.trainingSet = train_datagen.flow_from_directory(training_set_path,
                                                                 target_size = self.inputSize,
                                                                 batch_size = batch_size,
                                                                 class_mode = 'categorical')
            
            # Make the test set
            self.testSet = test_datagen.flow_from_directory(test_set_path,
                                                            target_size = self.inputSize,
                                                            batch_size = batch_size,
                                                            class_mode = 'categorical')
            
    
    def train_classifier(self, steps_per_epoch, epochs, validation_steps):
        print("Let's train!")
        # Train the classifier
        self.classifier.fit_generator(self.trainingSet,
                                      steps_per_epoch = steps_per_epoch,
                                      epochs = epochs,
                                      validation_data = self.testSet,
                                      validation_steps = validation_steps)
    
    def predict(self, img):
        # Preprocess the image
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        # predict if there is a cat or a bird.
        result = self.classifier.predict(img)
        # get the indices
        self.testSet.class_indices
        
        # Show the result for the one node output network
        if self.outputType == 1:
            # show prediction
            if result[0][0] == 1:
                prediction = 'bird'
            else:
                prediction = 'cat'
        
        # Show the result for the two node output network
        if self.outputType == 0:
            # find the class label index with the largest corresponding probability
            print("PREDICTIONS: ", result)
            result1 = round(result[0][0], 2)
            result2 = round(result[0][1], 2)
            
            # get the prediction
            if result1 > result2:
                prediction = 'bird'
            elif result1 < result2:
                prediction = 'cat'
            else:
                prediction = None
        
        return prediction
    
    def save_CNN_weights(self, path):
        self.classifier.save(path)



# This main is used to train the network and test if the functions work
if __name__ == '__main__':
    
    # parameter of the convolutional neural network
    first_convolutional_layer_args = select_1st_convolutional_layer_args(32, (3, 3), (64, 64, 3))
    second_convolutional_layer_args = select_2nd_convolutional_layer_args(32, (3, 3))
    first_pooling_layer_args = select_pooling_layer_args(2, 2)
    second_pooling_layer_args = select_pooling_layer_args(2, 2)
    hidden_layer_arg = select_hidden_layer_args(140)
    output_layer_arg1 = 1
    output_layer_arg2 = 0
    
    # Create the one node output neural network
    BvC1 = CNN_Classifier(first_convolutional_layer_args,
                          first_pooling_layer_args,
                          second_convolutional_layer_args,
                          second_pooling_layer_args,
                          hidden_layer_arg,
                          output_layer_arg1)
    
    # Create the one node output neural network
    BvC2 = CNN_Classifier(first_convolutional_layer_args,
                          first_pooling_layer_args,
                          second_convolutional_layer_args,
                          second_pooling_layer_args,
                          hidden_layer_arg,
                          output_layer_arg2)
    
    
    # making training & test sets for BvC 1
    BvC1.make_training_and_test_sets()
    # train the one node output network for BvC1
    BvC1.train_classifier(steps_per_epoch = 680, epochs = 40, validation_steps = 680)
    
    # making training & test sets for BvC2
    BvC2.make_training_and_test_sets()
    # train the one node output network for BvC2
    BvC2.train_classifier(steps_per_epoch = 680, epochs = 40, validation_steps = 680)
    
    
    # load a test image
    test_image = image.load_img('dataset/single_prediction/bc.jpeg', target_size = (64, 64))
    
    # test the prediction of the one node output network
    prediction1 = BvC1.predict(test_image)
    print(prediction1)
    if prediction1 == "bird":
        print("Correct")
    else:
        print("Incorrect")
    
    # test the prediction of the two node output network
    prediction2 = BvC2.predict(test_image)
    print(prediction2)
    
    if prediction2 == "bird":
        print("Correct")
    else:
        print("Incorrect")
    
    
    # Save the neural network weights for both networks
    BvC1.save_CNN_weights('/Users/TEB/IntuitiveSurgical/Interview/BirdsVsCats3.h5')
    BvC2.save_CNN_weights('/Users/TEB/IntuitiveSurgical/Interview/BirdsVsCats4.h5')
    
