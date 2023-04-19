import tensorflow as tf
import os
import numpy as np

from utils_.general_utils import confirm_directory
from utils_.model_architectures import ae_by_hassan, ae_by_hassan_gray, auto_encoded_model_by_atif
from utils_.attack.attack_utils import Attack

from utils_.custom_layers.quantization import Quantization, Variable_Quantization
from utils_.custom_layers.selective_convolution import Selective_Filter



preprocessing_layers = {
    'qusecnets': Quantization,
    'variable_qusecnets': Variable_Quantization,
    'sscnets': Selective_Filter
}


class Keras_Model:
    def __init__(self, path, data=None, 
                 preprocessing_parameters = None,
                 model=None, model_architecture=None, 
                 verbose=True):
        
        self.path = path
        
        self.data = data
        
        self.vanilla_model = model
        self.model_architecture = model_architecture
        
        self.verbose = verbose
        
    
    # Custom preprocessing layers - either QuSecNets or SSCNets
    def prepare_preprocessing_layer(self):
        self.preprocessing_layer = preprocessing_layers[self.preprocessing_parameters['layer_type']](
            quantization_levels = self.preprocessing_parameters['quantization_levels'],
            quantization_hardness = self.preprocessing_parameters['quantization_hardness'],
            filter_name = self.preprocessing_parameters['filter_name'],
            quantization_threshold = self.preprocessing_parameters['quantization_threshold']
        )
    
    
    def print_out(self, *print_statement, end=""):
        if self.verbose:
            print(*print_statement, end=end)
    
        
    def prepare_logits_model(self):
        self.logits_model = tf.keras.models.Model(inputs=self.model.input, 
                                                  outputs=self.model.get_layer("logits_layer").output)
        
        
    def secure_model(self):
        self.prepare_preprocessing_layer()
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=self.data.get_input_shape()))
        self.model.add(self.preprocessing_layer)
        self.model.add(self.vanilla_model)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def train_and_save(self, epochs=3, batch_size=64, model_name=None):
        
        self.model.fit(self.data.x_train, self.data.y_train,
                       epochs=epochs, batch_size=batch_size,
                       validation_data=(self.data.x_test, self.data.y_test))
        
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = self.data.data_name
        
        self.model.save(self.path+'models/'+self.model_name+'.h5')
        
        
    def compute_adversarial_images(self, N=100, attack_name='pgd',
                                      epsilon=0.01):
        attack = Attack(self.data, self)
        adversarial_images = attack.evaluate_on_attack(N, epsilon, attack_name=attack_name)
        self.model.evaluate(adversarial_images, self.data.y_test[:N])
        return adversarial_images
    
        
    def evaluate_secure_model(self, adversarial_images):
        
        clean_acc = self.secure_model.evaluate(self.data.x_test,
                                               self.data.y_test, 
                                               verbose=False)[1]
        adv_acc = self.secure_model.evaluate(adversarial_images, 
                                            self.data.y_test[:len(adversarial_images)], 
                                            verbose=False)[1]
        
        self.print_out("Accuracy on clean examples: ", clean_acc)
        self.print_out("Accuracy on perturbed adversarial examples: ", adv_acc)
                  
    
    def perform_adversarial_analysis(self, attack_name='pgd', epsilon=0.01):
        
        save_dir = 'adversarial_images/' + self.model_name + '/'
        save_name = 'adversarial_images_'+attack_name+'('+str(epsilon)+').npy'
        confirm_directory(self.path + save_dir)
        
        try:
            adversarial_images = np.load(self.path + save_dir + save_name)
            print(adversarial_images.shape)
        except:
            adversarial_images = self.compute_adversarial_images(attack_name=attack_name,
                                                                 epsilon=epsilon)
            print(adversarial_images.shape)    
            np.save(self.path + save_dir + save_name, adversarial_images)
        
        self.evaluate_secure_model(adversarial_images)
        
        
    def decision(self, x_in, type='vanilla'):
        if type == 'vanilla':
            return np.argmax(self.vanilla_model(x_in), axis=-1)
        else:
            return np.argmax(self.model(x_in), axis=-1)
    
    