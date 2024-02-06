""" Created: 29.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import math, sys, trimesh
import numpy as np
import tensorflow as tf

#==============================================================================
# Define a custom subclass to encapsulate tf.keras.metrics.Metric logic & state

# Inherits from 'tf.keras.metrics.Metric'

class MeanAbsoluteErrorMetric(tf.keras.metrics.Metric):
    
    # Initialise internal state variables using the 'self.add_weight()' method
    def __init__(self,name='mae_metric',**kwargs):
        
        super().__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name='error_sum',initializer='zeros')
        self.n_batches = self.add_weight(name='n_batches',initializer='zeros')
        return None

    # Define a method to update the state variables after each train minibatch 
    def update_state(self,true,pred):
                
        mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(pred,true)))
        self.error_sum.assign_add(mae)
        self.n_batches.assign_add(1.0)
        return None
    
    # Define a method to reset the state variables after each terminated epoch
    def reset_state(self):
        
        self.error_sum.assign(0.0)
        self.n_batches.assign(0.0)
        return None
    
    # Define a method to evaluate and return the mean squared error metric
    def result(self):
        
        mse = self.error_sum/self.n_batches
        return mse
    
    
#==============================================================================
# Define a console logger class to simultaneously log and print stdout messages

class Logger():
    
    # Initialise internal states and open log file
    def __init__(self, logfile):
        
        self.stdout = sys.stdout
        self.txtlog = open(logfile,'w')

    # Define a function to write to both stdout and txt
    def write(self, text):
        
        self.stdout.write(text)
        self.txtlog.write(text)
        self.txtlog.flush()
        
    # Define a function to flush both stdout and txt streams
    def flush(self):
        
        self.stdout.flush()
        self.txtlog.flush()

    # Define a function to close both stdout and txt streams
    def close(self):
        
        self.stdout.close()
        self.txtlog.close()
        

#==============================================================================
# Define a function to perform training on batches of data within the main loop

def TrainStep(model,optimiser,metric,sample_points_3d,signed_distances):
            
    # Open 'GradientTape' to record the operations run in each forward pass
    with tf.GradientTape() as tape:
        
        # Compute a forward pass on the current mini-batch
        signed_distances_predicted = model(sample_points_3d,training=True)
        
        # Compute the mean-squared error for the current mini-batch
        mae = MeanAbsoluteError(signed_distances,signed_distances_predicted)
    ##
   
    # Determine the weight and bias gradients with respect to error
    gradients = tape.gradient(mae,model.trainable_variables)
    
    # Update the weight and bias values to minimise the total error
    optimiser.apply_gradients(zip(gradients,model.trainable_variables))
            
    # Update the training metric
    metric.update_state(signed_distances,signed_distances_predicted)
        
    return None

#==============================================================================
# Define a function that computes the mean squared loss on predictions 

def MeanAbsoluteError(true,pred):
        
    # Compute the weighted mean squared error between signals
    mae = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(pred,true)))                           
    
    return mae

#==============================================================================
# Define a function to calculate the current learning rate based on epoch/decay

def GetLearningRate(initial_lr,half_life,epoch):
    
    # Decay learning rate following an exponentially decaying curve
    current_lr = initial_lr / (2**(epoch//half_life))

    return current_lr

#==============================================================================