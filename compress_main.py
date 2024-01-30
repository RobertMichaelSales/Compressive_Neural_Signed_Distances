""" Created: 29.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os, time, json, math, psutil, sys, gc, datetime, trimesh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.enable()

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from data_management         import MakeDatasetFromGenerator,MakeDatasetFromTensorSlice
from network_model           import ConstructNetwork
from configuration_classes   import GenericConfigurationClass
from compress_utilities      import TrainStep,GetLearningRate,MeanAbsoluteErrorMetric,Logger

#==============================================================================

def compress(network_config,dataset_config,runtime_config,training_config,o_filepath):
        
    print("-"*80,"\nSQUASHSDF: IMPLICIT NEURAL COMPRESSIVE SIGNED DISTANCE FUNCTIONS (by Rob Sales)")
    
    print("\nDateTime: {}".format(datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S")))
    
    #==========================================================================
    # Check whether hardware acceleration is enabled
    print("-"*80,"\nCHECKING SYSTEM REQUIREMENTS:")
    
    print("\n{:30}{}".format("TensorFlow Version:",tf.__version__))
    gpus = tf.config.list_physical_devices('GPU')
    
    if (len(gpus) != 0): 
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Enabled") )
        tf.config.experimental.set_memory_growth(gpus[0],True)
    else:        
        print("\n{:30}{} - {}".format("CUDA GPU(s) Detected:",len(gpus),"Hardware Acceleration Is Disabled"))
        raise SystemError("GPU device not found. Try restarting your system to resolve this error.")
    ##
        
    #==========================================================================
    # Check whether the input size exceeds available memory
    print("-"*80,"\nCHECKING MEMORY REQUIREMENTS:")
    
    # Get and display available memory (hardware limit)
    available_memory = psutil.virtual_memory().available
    print("\n{:30}{:.3f} GigaBytes".format("Available Memory:",(available_memory/1e9)))
    
    # Set and display threshold memory (software limit)
    threshold_memory = int(20*1e9)
    print("\n{:30}{:.3f} GigaBytes".format("Threshold Memory:",(threshold_memory/1e9)))
    
    # Get and display input file size
    input_file_size = os.path.getsize(dataset_config.i_filepath)
    print("\n{:30}{:06.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    # Determine whether data exceeds memory and choose how to load the dataset
    dataset_exceeds_memory = (input_file_size > min(available_memory,threshold_memory))

    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")

    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",dataset_config.i_filepath.split("/")[-1]))
    mesh = trimesh.load(dataset_config.i_filepath)
    
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight. SDFs can only be calculated for watertight meshes.")
    ##    

    # Generate a dataset to supply coordinates and signed distances in training 
    dataset = MakeDatasetFromGenerator(mesh=mesh,batch_size=training_config.batch_size,sample_method=training_config.sample_method,dataset_size=training_config.dataset_size)         
    print("\n{:30}{}".format("Method:","MakeDatasetFromGenerator()"))
    
    #==========================================================================
    # Configure network 
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    network_config.neurons_per_layer = 32 # Set this as a method once I have decided what the network is actually going to look like 
    
    # Build SquashSDF from the network configuration information
    SquashSDF = ConstructNetwork(hidden_layers=network_config.hidden_layers,neurons_per_layer=network_config.neurons_per_layer,activation=network_config.activation)
                      
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric (custom weighted mean-squared error metric)
    metric = MeanAbsoluteErrorMetric()
        
    # Load the training step function as a tf.function for speed increases
    TrainStepTFF = tf.function(TrainStep)
        
    # Save an image of the network graph (helpful to check)
    # tf.keras.utils.plot_model(model=SquashSDF,to_file=os.path.join(o_filepath,"network_graph.png"))     
       
#==============================================================================
# Define the main function to run when file is invoked from within the terminal

if __name__=="__main__":
    
        network_config  = GenericConfigurationClass({"network_name" : "squashsdf", "hidden_layers" : 8, "target_compression_ratio" : 10, "minimum_neurons_per_layer" : 1, "activation" : "elu",})
    
        dataset_config  = GenericConfigurationClass({"i_filepath" : "/home/rms221/Documents/Compressive_Signed_Distance_Functions/My Attempt/Meshes/bumpy-cube.obj",})
    
        runtime_config  = GenericConfigurationClass()
       
        training_config = GenericConfigurationClass({"initial_lr" : 0.001, "batch_size" : 1024, "epochs" : 30, "half_life" : 2, "dataset_size" : 1e5, "sample_method" : "vertice"})
        
        o_filepath      = ""
        
        compress(network_config=network_config,dataset_config=dataset_config,runtime_config=runtime_config,training_config=training_config,o_filepath=o_filepath)
         
else: pass

#==============================================================================
