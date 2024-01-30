""" Created: 29.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np
import tensorflow as tf

#==============================================================================

def ConstructNetwork(hidden_layers,neurons_per_layer,activation):

    for layer in range(hidden_layers):
        
        if (layer == 0):
            
            inputs = tf.keras.layers.Input(shape=(3,),name="L{}_input".format(layer))
            x = tf.keras.layers.Dense(units=neurons_per_layer,activation=activation,name="L{}_dense".format(layer))(inputs)
            
        else:
            
            x = tf.keras.layers.Dense(units=neurons_per_layer,activation=activation,name="L{}_dense".format(layer))(x)
    
        ##
        
        outputs = tf.keras.layers.Dense(units=1,activation="tanh",name="L{}_output".format(hidden_layers))(x)
        
    ##
    
    SquashSDF = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    return SquashSDF

##

#==============================================================================