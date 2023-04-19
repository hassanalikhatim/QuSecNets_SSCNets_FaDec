from tensorflow.keras import utils
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image



class Dataset:
    def __init__(self, data_name='mnist'):
        self.data_name = data_name
    
    
    def get_input_shape(self):
        return self.x_test.shape[1:]
    
    
    def get_output_shape(self):
        return self.y_test.shape[1:]
    
    
    def get_class_names(self):
        class_names = {
            'cifar10': ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'],
            'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        }
        return class_names[self.data_name]
    
    
    def load_and_prepare_data(self):
        data_loader = {
            "mnist": tf.keras.datasets.mnist.load_data,
            "cifar10": tf.keras.datasets.cifar10.load_data,
        }
        (x_train, y_train), (x_test, y_test) = data_loader[self.data_name]()
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        x_train = x_train.astype('float32')/255.
        x_test = x_test.astype('float32')/255.
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], x_train.shape[2], -1))
        
        num_classes = np.max(y_train) + 1
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test
    
    