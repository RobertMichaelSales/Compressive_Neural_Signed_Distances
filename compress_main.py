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

from data_management         import LoadMeshDataset,MakeMeshDataset,LoadGridDataset,MakeGridDataset
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
    threshold_memory = int(16*1e9)
    print("\n{:30}{:.3f} GigaBytes".format("Threshold Memory:",(threshold_memory/1e9)))
    
    # Get and display input file size
    input_file_size = os.path.getsize(dataset_config.mesh_filepath)
    print("\n{:30}{:06.3f} GigaBytes".format("Input File Size:",(input_file_size/1e9)))
    
    # Determine whether data exceeds memory and choose how to load the dataset
    dataset_exceeds_memory = (input_file_size > min(available_memory,threshold_memory))

    #==========================================================================
    # Initialise i/o 
    print("-"*80,"\nINITIALISING INPUTS:")

    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",dataset_config.mesh_filepath.split("/")[-1]))
    mesh = trimesh.load(dataset_config.mesh_filepath)
    
    # # Make the validation grid 
    # print("\n{:30}{}".format("Validation Resolution:",training_config.grid_resolution))
    # grid = ValidationGrid(mesh=mesh,resolution=training_config.grid_resolution)
    
    # Check if the mesh is 'watertight' for computing SDFs
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight. SDFs can only be calculated for watertight meshes.")
    ##    
    
    #==========================================================================
    # Configure dataset
    print("-"*80,"\nCONFIGURING DATASET:")
    
    base_data_path = os.path.splitext(dataset_config.mesh_filepath)[0]
    mesh_data_path = os.path.join(base_data_path,"mesh_"+str(training_config.sample_method)+"_"+str(training_config.dataset_size)+".npy")
    grid_data_path = os.path.join(base_data_path,"grid_"+str(training_config.grid_resolution)+"_"+str(training_config.bbox_scale)+".npy")
    
    show = runtime_config.visualise_mesh_dataset
        
    # Make mesh dataset to supply coordinates and signed distances 
    if os.path.exists(mesh_data_path):
        print("\n{:30}{}".format("Loading Mesh Dataset:",mesh_data_path.split("/")[-1]),end="    ")
        mesh_dataset = LoadMeshDataset(mesh=mesh,batch_size=training_config.batch_size,sample_method=training_config.sample_method,dataset_size=training_config.dataset_size,load_filepath=mesh_data_path,show=show)
        precomputed_mesh = True
        print("\n\n{:30}{}".format("dataset_size:",mesh_dataset.dataset_size))
        print("\n{:30}{}".format("batch_size:",mesh_dataset.batch_size))
    else:
        print("\n{:30}{}".format("Forming Mesh Dataset:",mesh_data_path.split("/")[-1]),end="\n\n")
        mesh_dataset = MakeMeshDataset(mesh=mesh,batch_size=training_config.batch_size,sample_method=training_config.sample_method,dataset_size=training_config.dataset_size,save_filepath=mesh_data_path,show=show)        
        precomputed_mesh = False 
        print("\n\n{:30}{}".format("dataset_size:",mesh_dataset.dataset_size))
        print("\n{:30}{}".format("batch_size:",mesh_dataset.batch_size))
    ##
    
    show = runtime_config.visualise_grid_dataset
               
    # Make grid dataset to supply coordinates and signed distances 
    if os.path.exists(grid_data_path):
        print("\n{:30}{}".format("Loading Grid Dataset:",grid_data_path.split("/")[-1]),end="    ")
        grid_dataset = LoadGridDataset(mesh=mesh,batch_size=training_config.batch_size,sample_method=training_config.sample_method,dataset_size=training_config.dataset_size,load_filepath=grid_data_path,show=show)
        precomputed_grid = True
        print("\n\n{:30}{}".format("grid_resolution:",grid_dataset.resolution))
        print("\n{:30}{}".format("batch_size:",grid_dataset.batch_size))
    else:
        print("\n{:30}{}".format("Forming Grid Dataset:",grid_data_path.split("/")[-1]),end="\n\n")
        grid_dataset = MakeMeshDataset(mesh=mesh,batch_size=training_config.batch_size,sample_method=training_config.sample_method,dataset_size=training_config.dataset_size,save_filepath=grid_data_path,show=show)        
        precomputed_grid = False 
        print("\n\n{:30}{}".format("grid_resolution:",grid_dataset.resolution))
        print("\n{:30}{}".format("batch_size:",grid_dataset.batch_size))
    ##    
    
    return None
    
    #==========================================================================
    # Configure network 
    print("-"*80,"\nCONFIGURING NETWORK:")
    
    # Set this as a method once I have decided what the network is actually going to look like 
    network_config.neurons_per_layer = 32 
    
    # Build SquashSDF from the network configuration information
    SquashSDF = ConstructNetwork(hidden_layers=network_config.hidden_layers,neurons_per_layer=network_config.neurons_per_layer,activation=network_config.activation)
                      
    # Set a training optimiser
    optimiser = tf.keras.optimizers.Adam()
    
    # Set a performance metric (custom weighted mean-squared error metric)
    metric = MeanAbsoluteErrorMetric()
        
    # Load the training step function as a tf.function for speed increases
    TrainStepTFF = tf.function(TrainStep)
        
    # Save an image of the network graph (helpful to check)
    tf.keras.utils.plot_model(model=SquashSDF,to_file=os.path.join(o_filepath,"network_graph.png"))     
        
    return None
    
    #==========================================================================
    # Training loop
    print("-"*80,"\nCOMPRESSING MESH:")
        
    # Create a dictionary of lists to store training data
    training_data = {"epoch":[],"training_error":[],"time":[],"learning_rate":[],}

    # Start the overall training timer
    training_time_tick = time.time()
    
    # Iterate through each epoch
    for epoch in range(training_config.epochs):
        
        print("\n",end="")
                        
        # Store and print the current epoch number
        training_data["epoch"].append(float(epoch))
    
        # Determine, update, store and print the learning rate 
        learning_rate = GetLearningRate(initial_lr=training_config.initial_lr,half_life=training_config.half_life,epoch=epoch)
        optimiser.lr.assign(learning_rate)
        training_data["learning_rate"].append(float(learning_rate))   
        print("{:30}{:.3E}".format("Learning rate:",learning_rate))
        
        # Start timing current epoch
        epoch_time_tick = time.time()
        
        # Iterate through each batch
        for batch, (sample_points_3d,signed_distances) in enumerate(mesh_dataset):
            
            # Print the current batch number and run a training step
            if runtime_config.print_verbose: print("\r{:30}{:04}/{:04}".format("Batch number:",(batch+1),mesh_dataset.size),end="") 
            TrainStepTFF(model=SquashSDF,optimiser=optimiser,metric=metric,sample_points_3d=sample_points_3d,signed_distances=signed_distances)

            if batch >= mesh_dataset.size: break
            
        ##
        
        print("\n",end="")
        
        # End the epoch time and store the elapsed time 
        epoch_time_tock = time.time() 
        epoch_time = float(epoch_time_tock-epoch_time_tick)
        training_data["time"].append(epoch_time)
        print("{:30}{:.2f} seconds".format("Epoch time:",epoch_time))
        
        # Fetch, store and reset and the training error
        error = float(metric.result().numpy())
        metric.reset_states()
        training_data["training_error"].append(error)
        print("{:30}{:.7f}".format("Mean absolute error:",error))     
        
        # Early stopping for diverging training results
        if np.isnan(error): 
            print("{:30}{:}".format("Early stopping:","Error has diverged")) 
            runtime_config.save_network_flag = False
            runtime_config.save_outputs_flag = False
            break
        else: pass
    
    ##   
 
    # End the overall training timer
    training_time_tock = time.time()
    training_time = float(training_time_tock-training_time_tick)
    print("\n{:30}{:.2f} seconds".format("Training duration:",training_time))        
    
    return None
    
    #==========================================================================
    # Finalise outputs    

    # Generate the validation SDF 
    ## o_values.flat = SquashNet.predict(o_volume.flat,batch_size=training_config.batch_size,verbose="1")                                                   ?????? <<<<<<    
    ## o_values.data = np.reshape(o_values.flat,(o_volume.data.shape[:-1]+(o_values.dimensions,)),order="C")                                                ?????? <<<<<<    
    # Compute the validation loss
    ## print("{:30}{:.3f}".format("Mean absolute error:","NEEDS IMPLIMENTING"))                                                                             ?????? <<<<<<
    ## training_data["validation_error"].append("NEEDS IMPLIMENTING")                                                                                       ?????? <<<<<<
    
    # Pack the configuration dictionaries into just one
    combined_config_dict = (network_config | training_config | runtime_config | dataset_config)
    
    #==========================================================================
    # Save network
    
    if runtime_config.save_network_flag:
        print("-"*80,"\nSAVING NETWORK:")
        print("\n",end="")
        
        # Save the parameters
        parameters_path = os.path.join(o_filepath,"parameters.bin")
        # EncodeParameters(network=SquashNet,parameters_path=parameters_path,values_bounds=(i_values.max,i_values.min))                                     ?????? <<<<<<
        print("{:30}{}".format("Saved parameters to:",parameters_path.split("/")[-1]))
        
        # Save the architecture
        architecture_path = os.path.join(o_filepath,"architecture.bin")
        # EncodeArchitecture(layer_dimensions=network_config.layer_dimensions,frequencies=network_config.frequencies,architecture_path=architecture_path)   ?????? <<<<<<
        print("{:30}{}".format("Saved architecture to:",architecture_path.split("/")[-1]))
    else: pass

    #==========================================================================
    # Save outputs
    
    if runtime_config.save_outputs_flag:
        print("-"*80,"\nSAVING OUTPUTS:")
        print("\n",end="")
        
        # Save i_volume and i_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"i_volume")
        # SaveData(output_data_path=output_data_path,volume=i_volume,values=i_values,reverse_normalise=True)                                                ?????? <<<<<<
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1])) 
        
        # Save o_volume and o_values to ".npy" and ".vtk" files
        output_data_path = os.path.join(o_filepath,"o_volume")
        # SaveData(output_data_path=output_data_path,volume=o_volume,values=o_values,reverse_normalise=True)                                                ?????? <<<<<<
        print("{:30}{}.{{npy,vts}}".format("Saved output files as:",output_data_path.split("/")[-1]))        
    else: pass    
    
    #==========================================================================
    # Save results
    
    if runtime_config.save_results_flag:
        print("-"*80,"\nSAVING RESULTS:")        
        print("\n",end="")
        
        # Save the training data
        training_data_path = os.path.join(o_filepath,"training_data.json")
        with open(training_data_path,"w") as file: json.dump(training_data,file,indent=4,sort_keys=True)
        print("{:30}{}".format("Saved training data to:",training_data_path.split("/")[-1]))
    
        # Save the configuration
        combined_config_path = os.path.join(o_filepath,"config.json")
        with open(combined_config_path,"w") as file: json.dump(combined_config_dict,file,indent=4)
        print("{:30}{}".format("Saved configuration to:",combined_config_path.split("/")[-1]))
    else: pass
    
    #==========================================================================
    print("-"*80,"\n")
    
    
    return None
       
#==============================================================================
# Define the main function to run when file is invoked from within the terminal

if __name__=="__main__":
    
        network_config  = GenericConfigurationClass({"network_name" : "squashsdf", "hidden_layers" : 8, "target_compression_ratio" : 10, "minimum_neurons_per_layer" : 1, "activation" : "elu",})
    
        dataset_config  = GenericConfigurationClass({"mesh_filepath" : "/home/rms221/Documents/Compressive_Neural_Signed_Distances/inputs/bumpy-cube.obj"})
    
        runtime_config  = GenericConfigurationClass({"print_verbose" : True, "save_network_flag" : False, "visualise_mesh_dataset" : True})
       
        training_config = GenericConfigurationClass({"initial_lr" : 0.001, "batch_size" : 1024, "epochs" : 30, "half_life" : 2, "dataset_size" : 1000000, "sample_method" : "vertice", "grid_resolution" : 32, "bbox_scale" : 1.0})
        
        o_filepath      = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/outputs/"
        
        compress(network_config=network_config,dataset_config=dataset_config,runtime_config=runtime_config,training_config=training_config,o_filepath=o_filepath)
         
else: pass

#==============================================================================
