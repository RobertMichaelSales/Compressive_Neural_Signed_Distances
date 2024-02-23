""" Created: 06.02.2024  \\  Updated: 07.02.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries

import os, json, glob
import numpy as np

#==============================================================================

if __name__=="__main__": 

    # Set input data config options
    input_dataset_config_paths = sorted(glob.glob("/Data/SDF_Compression_Datasets/turbostream_*/turbostream_*_config.json"))
            
    # Set experiment number
    experiment_num = 3
    
    # Set counter and total
    count = 1
    total = len(input_dataset_config_paths)*1*1*1*1*1*1*1*1*2
    
    # Iterate through all inputs
    for input_dataset_config_path in input_dataset_config_paths:
    
        for compression_ratio in np.array([10]):
            
                for learning_rate in np.array([1e-3]):
                    
                    for batch_size in np.array([1024]):
                        
                        for frequencies in np.array([10]):
                                                                        
                            for hidden_layers in np.array([8]):
                                
                                for activation in np.array(["elu"]):
                                    
                                    for bits_per_neuron in np.array([32]):
                                        
                                        for normalise in np.array([True]):
                                            
                                            for network_architecture in np.array(["SIREN","BASIC"]):
        
                                                # Set experiment campaign name
                                                campaign_name = "EXP({:03d})_TCR({:011.6f})_ILR({:11.9f})_PEF({:03d})_NHL({:03d})_ACT({:})_BPN({:03d})_{:}_{:}".format(experiment_num,compression_ratio,learning_rate,frequencies,hidden_layers,activation,bits_per_neuron,network_architecture,"NORM" if normalise else "ORIG") 
                                                                                
                                                # Print this experiment number
                                                print("\n");print("*"*80);print("Experiment {}/{}: '{}'".format(count,total,campaign_name));print("*"*80);print("\n")
                                                
                                                # Define the dataset config
                                                with open(input_dataset_config_path) as input_dataset_config_file: dataset_config = json.load(input_dataset_config_file)
                                                
                                                # Define the network config
                                                network_config = {
                                                    "network_name"              : str(campaign_name),
                                                    "hidden_layers"             : int(hidden_layers),
                                                    "frequencies"               : int(frequencies),
                                                    "target_compression_ratio"  : float(compression_ratio),
                                                    "activation"                : str("elu"),
                                                    "minimum_neurons_per_layer" : int(1),
                                                    "bits_per_neuron"           : int(bits_per_neuron),
                                                    "normalise"                 : bool(normalise),
                                                    "network_architecture"      : str(network_architecture)
                                                    }
                                                                       
                                                # Define the runtime config
                                                runtime_config = {
                                                    "print_verbose"             : bool(False),
                                                    "save_network_flag"         : bool(False),
                                                    "save_outputs_flag"         : bool(True),
                                                    "save_results_flag"         : bool(True),
                                                    "visualise_mesh_dataset"    : bool(False),
                                                    "visualise_grid_dataset"    : bool(False),
                                                    }
                                                
                                                # Define the training config
                                                training_config = {
                                                    "initial_lr"                : float(learning_rate),
                                                    "batch_size"                : int(batch_size),
                                                    "epochs"                    : int(100),
                                                    "half_life"                 : int(5),     
                                                    "dataset_size"              : int(1e6),
                                                    "sample_method"             : str("vertice"),
                                                    "grid_resolution"           : int(128),
                                                    "bbox_scale"                : float(1.1),
                                                    }            
                                                
                                                # Define the output directory
                                                o_filepath = "/Data/SDF_Compression_Experiments/" + os.path.join(*dataset_config["mesh_filepath"].split("/")[-1:]).replace(".obj","")
                                                                                
                                                # Run the compression experiment
                                                runstring = "python SourceCode/compress_main.py " + "'" + json.dumps(network_config) + "' '" + json.dumps(dataset_config) + "' '" + json.dumps(runtime_config) + "' '" + json.dumps(training_config) + "' '" + o_filepath + "'"
                                                os.system(runstring)
                                                
                                                # Define the plotting config
                                                plotting_config = {
                                                    "filepath"                  : os.path.join(o_filepath,campaign_name),
                                                    "render_zoom"               : float(1.0),                           
                                                    }  
                                                
                                                # Render the results in ParaView
                                                runstring = "pvpython ParaView/render.py " + "'" + json.dumps(plotting_config) + "'"
                                                os.system(runstring)
                                                
                                                count = count + 1
                                        
                                        ##
                                    ##                                    
                                ##                                
                            ##
                        ##
                    ##
                ##
            ##
        ##
    ##
##
        
#==============================================================================
   
