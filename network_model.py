""" Created: 29.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================

def ConstructNetwork(layer_dimensions,activation):

    import random; tf.random.set_seed(123);np.random.seed(123);random.seed(123)
    
    total_layers = len(layer_dimensions)
    
    for layer in range(total_layers):
        
        if (layer == 0):
            
            inputs = tf.keras.layers.Input(shape=(layer_dimensions[layer],),name="L{}_input".format(layer))
            x = tf.keras.layers.Dense(units=layer_dimensions[layer+1],activation=activation,name="L{}_dense".format(layer))(inputs)
            
        elif (layer == (total_layers - 1)):
            
            outputs = tf.keras.layers.Dense(units=layer_dimensions[layer],activation="tanh",name="L{}_output".format(layer))(x)
            
        else:        
        
            x = tf.keras.layers.Dense(units=layer_dimensions[layer],activation=activation,name="L{}_dense".format(layer))(x)
    
        ##
        
    ##
    
    SquashSDF = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    return SquashSDF

##

#==============================================================================