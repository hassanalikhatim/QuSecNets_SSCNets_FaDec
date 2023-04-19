import tensorflow as tf



class Selective_Filter(tf.keras.layers.Layer):
    
    def __init__(self, filter_name='filter1', 
                 quantization_threshold=0.2, quantization_hardness=500, 
                 **kwargs):
        
        self.filter_name = filter_name
        self.quantization_threshold = quantization_threshold
        self.quantization_hardness = quantization_hardness
        
        filter1 = [
            [   0,-1/4,   0],
            [-1/4,   1,-1/4],
            [   0,-1/4,   0]
        ]
        filter2 = [
            [-1/8,-1/8,-1/8],
            [-1/8,   1,-1/8],
            [-1/8,-1/8,-1/8]
        ]
        filter3 = [
            [  -1,  -2,   1],
            [   0,   0,   0],
            [   1,   2,   1]
        ]
        filter4 = [
            [  -1,  -1,   1],
            [   0,   0,   0],
            [   1,   1,   1]
        ]
        self.filters = {
            'filter1': filter1,
            'filter2': filter2,
            'filter3': filter3,
            'filter4': filter4
        }
        
        super(Selective_Filter, self).__init__()
        
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.filter = self.filters[self.filter_name]
        return super().build(input_shape)
        
        
    def apply_filter(self, filterRR, image):
        filterR = tf.cast(tf.reshape(filterRR, [3, 3, self.channel, 1]), tf.float32)
        imageGQEdgedTensor=tf.nn.conv2d(image, filterR, strides=[1, 1, 1, 1], padding='SAME')
        return tf.abs(imageGQEdgedTensor)
    
    
    def sigmoid_fn(self, x_in):
        return 1/(1+tf.exp(-x_in))
    
    
    def call(self, inputs):
        
        grayTensor = tf.abs(self.apply_filter(self.filter, inputs))
        grayTensor = self.sigmoid_fn(grayTensor - self.quantization_threshold)
        grayTensor = tf.multiply(inputs, grayTensor)
        
        return grayTensor