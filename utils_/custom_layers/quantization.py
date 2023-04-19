import tensorflow as tf
    


class Quantization(tf.keras.layers.Layer):
    
    def __init__(self, quantization_levels=16, quantization_hardness=1, **kwargs):
        
        self.quantization_hardness = quantization_hardness
        
        self.quantization_levels = quantization_levels
        self.quantization_thresholds_interval = 1 / quantization_levels
        self.quantization_thresholds = [ (i+1)*self.quantization_thresholds_interval for i in range(quantization_levels)]
        
        super(Quantization, self).__init__()
        
        
    def sigmoid_fn(self, x_in):
        return 1/(1+tf.exp(-x_in))
        
        
    def call(self, inputs):
        
        output = 0 * inputs
        for quantization_threshold in self.quantization_thresholds:
            output += self.sigmoid_fn(
                self.quantization_hardness * (inputs - quantization_threshold)
            )
        
        output /= (self.quantization_levels - 1)
        
        return tf.clip_by_value(output, 0, 1)
    

  
class Variable_Quantization(tf.keras.layers.Layer):
    
    def __init__(self, quantization_levels=16, quantization_hardness=1, **kwargs):
        
        self.quantization_hardness = quantization_hardness
        
        self.quantization_levels = quantization_levels
        self.quantization_thresholds_interval = 1 / quantization_levels
        
        quantization_thresholds = [(i+1)*self.quantization_thresholds_interval for i in range(quantization_levels)]
        self.variable_quantization_thresholds = tf.Variable(
            initial_value=tf.convert_to_tensor(quantization_thresholds),
            trainable=True
        )
        
        super(Variable_Quantization, self).__init__()
        
        
    def sigmoid_fn(self, x_in):
        return 1/(1+tf.exp(-x_in))
        
        
    def call(self, inputs):
        
        output = 0 * inputs
        for quantization_threshold in self.variable_quantization_thresholds:
            output += self.sigmoid_fn(
                self.quantization_hardness * (inputs - quantization_threshold)
            )
        
        output /= (self.quantization_levels - 1)
        
        return tf.clip_by_value(output, 0, 1)
    
    
    
