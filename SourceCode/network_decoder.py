""" Created: 06.02.2024  \\  Updated: 14.03.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np

#==============================================================================
# Define a function to decode the network layer dimensions (architecture) from

def DecodeArchitectureBASIC(architecture_path):
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1]).astype('uint16').tobytes())

    # Open the architecture file in 'read as binary' mode
    with open(architecture_path,"rb") as file:
        
        # Read the total number of layer dimensions as bytestring
        total_num_layers_as_bytestring = file.read(1*bytes_per_value)
        total_num_layers = int(np.frombuffer(total_num_layers_as_bytestring,dtype='uint16'))
        
        # Read the list of network layer dimensions as bytestring
        layer_dimensions_as_bytestring = file.read(total_num_layers*bytes_per_value)
        layer_dimensions = list(np.frombuffer(layer_dimensions_as_bytestring,dtype=np.uint16))
        
        # Read the number of positional encoding frequencies as bytestring
        frequencies_as_bytestring = file.read(1*bytes_per_value)
        frequencies = int(np.frombuffer(frequencies_as_bytestring,dtype='uint16'))
        
        '''
        Decode activation, and remove assignment below
        '''
        
        activation = "relu"
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return layer_dimensions,frequencies,activation

#==============================================================================
# Define a function to decode the network layer dimensions (architecture) from

def DecodeArchitectureSIREN(architecture_path):
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1]).astype('uint16').tobytes())

    # Open the architecture file in 'read as binary' mode
    with open(architecture_path,"rb") as file:
        
        # Read the total number of layer dimensions as bytestring
        total_num_layers_as_bytestring = file.read(1*bytes_per_value)
        total_num_layers = int(np.frombuffer(total_num_layers_as_bytestring,dtype='uint16'))
        
        # Read the list of network layer dimensions as bytestring
        layer_dimensions_as_bytestring = file.read(total_num_layers*bytes_per_value)
        layer_dimensions = list(np.frombuffer(layer_dimensions_as_bytestring,dtype=np.uint16))
        
        # Read the number of positional encoding frequencies as bytestring
        frequencies_as_bytestring = file.read(1*bytes_per_value)
        frequencies = int(np.frombuffer(frequencies_as_bytestring,dtype='uint16'))
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return layer_dimensions,frequencies

#==============================================================================
# Define a function to decode the network layer dimensions (architecture) from

def DecodeArchitectureGAUSS(architecture_path):
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1]).astype('uint16').tobytes())

    # Open the architecture file in 'read as binary' mode
    with open(architecture_path,"rb") as file:
        
        # Read the total number of layer dimensions as bytestring
        total_num_layers_as_bytestring = file.read(1*bytes_per_value)
        total_num_layers = int(np.frombuffer(total_num_layers_as_bytestring,dtype='uint16'))
        
        # Read the list of network layer dimensions as bytestring
        layer_dimensions_as_bytestring = file.read(total_num_layers*bytes_per_value)
        layer_dimensions = list(np.frombuffer(layer_dimensions_as_bytestring,dtype=np.uint16))
        
        '''
        Decode activation, and remove assignment below
        Decode gaussian_kernel and remove assigment below
        '''
        
        activation = "relu"
        gaussian_kernel = np.ones((1,3))
        
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return layer_dimensions,activation,gaussian_kernel

#==============================================================================
# Define a function to decode the weights/biases of each layer

def DecodeParameters(network,network_architecture,parameters_path):
    
    # If BASIC then sort layers using SortLayerNamesBASIC
    if (network_architecture == "basic"): SortLayerNames = SortLayerNamesBASIC
    
    # If SIREN then sort layers using SortLayerNamesBASIC
    if (network_architecture == "siren"): SortLayerNames = SortLayerNamesSIREN
    
    # If GAUSS then sort layers using SortLayerNamesBASIC
    if (network_architecture == "gauss"): SortLayerNames = SortLayerNamesGAUSS
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = network.get_weight_paths().keys()
    layer_names = sorted(list(layer_names),key=SortLayerNames)
    
    # Determine the number of bytes per value
    bytes_per_value = len(np.array([1.0]).astype('float32').tobytes())
    
    # Open the weights file in 'read as binary' mode
    with open(parameters_path,"rb") as file:
    
        # Create an empty dictionary of the form {layer_name,weights}
        parameters = {}
        
        # Iterate through each of the network layers
        for layer_name in layer_names:
            
            # Extract the un-initialised layer from the network
            layer = network.get_weight_paths()[layer_name].numpy()
            
            # Read the current layer weights bytestring
            weights_as_bytestring = file.read(layer.size*bytes_per_value)    
            
            # Convert the bytestring into a 1-d array
            weights = np.frombuffer(weights_as_bytestring,dtype='float32')
            
            # Resize the 1-d array according to layer.shape
            weights = np.reshape(weights,layer.shape,order="C")
            
            # Add the weights to the dictionary
            parameters[layer_name] = weights
        
        ##
        
        # Read the original centre bytestring
        original_centre_as_bytestring = file.read(bytes_per_value*3)

        # Convert the bytestring into a 1-d array
        original_centre = np.frombuffer(original_centre_as_bytestring,dtype='float32')
        
        # Read the original centre bytestring
        original_radius_as_bytestring = file.read(bytes_per_value*1)

        # Convert the bytestring into a 1-d array
        original_radius = np.frombuffer(original_radius_as_bytestring,dtype='float32')
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()   
    ##
    
    return parameters,original_centre,original_radius

#==============================================================================
# Define a function to assign the weights/biases of each layer

def AssignParameters(network,parameters):
        
    # Iterate through each of the network layers
    for layer_name in parameters.keys():
        
        # Extract the un-initialised layer from the network
        layer = network.get_weight_paths()[layer_name]
                     
        # Assign the weights to the un-initialised network
        layer.assign(parameters[layer_name])
    ##
    
    return None
    
#==============================================================================
# Define a function to sort the layer names alpha-numerically so that the saved
# weights are always in the correct order

def SortLayerNamesBASIC(layer_name):
    
    layer_index = int(layer_name.split("_")[0][1:])

    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.50   
    
    return layer_index

#==============================================================================
# Define a function to sort the layer names alpha-numerically so that the saved
# weights are always in the same correct order (the as-constructed ordering)

def SortLayerNamesSIREN(layer_name):
    
    layer_index = int(layer_name.split("_")[0][1:])

    if "_a" in layer_name: 
        layer_index = layer_index
    
    if "_b" in layer_name: 
        layer_index = layer_index + 0.50
        
    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.25   

    return layer_index

#==============================================================================
# Define a function to sort the layer names alpha-numerically so that the saved
# weights are always in the correct order

def SortLayerNamesGAUSS(layer_name):
    
    layer_index = int(layer_name.split("_")[0][1:])

    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.50   
    
    return layer_index

#==============================================================================