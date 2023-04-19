import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, InputLayer
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Reshape



WEIGHT_DECAY = 1e-2


def cnn_model(data,
              regularization_constant=WEIGHT_DECAY, 
              n_layers=2, 
              activation_name='relu',
              compile_model=True):
    
    regularization_constant = WEIGHT_DECAY
    
    encoder = Sequential()
    encoder.add(InputLayer(shape=data.get_input_shape()))
    
    for l in range(n_layers):
        encoder.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(regularization_constant)))
        encoder.add(Activation(activation_name))
        encoder.add(BatchNormalization())
    
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Dropout(0.2))
    encoder.add(Flatten())
    
    encoder.add(Dense(data.get_output_shape()[0]), name='logits_layer')
    encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                        metrics=['accuracy'])
    
    return encoder

    