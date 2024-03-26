""" Created: 06.02.2024  \\  Updated: 14.03.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np

#==============================================================================
# Define a function to encode the network layer dimensions (or architecture) as
# a binary file containing strings of bytes

# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack'

def EncodeArchitecture(layer_dimensions,frequencies,architecture_path):

    # Extract the total number of layer dimensions to bytestrings
    total_num_layers = np.array(len(layer_dimensions)).astype('uint16')    
    total_num_layers_as_bytestring = total_num_layers.tobytes()
    
    # Extract the list of network layer dimensions to bytestrings
    layer_dimensions = np.array(layer_dimensions).astype('uint16')
    layer_dimensions_as_bytestring = layer_dimensions.tobytes()
    
    # Extract the number of positional encoding frequencies to bytestrings
    frequencies = np.array(frequencies).astype('uint16')
    frequencies_as_bytestring = frequencies.tobytes()
    
    # Open the architecture file in 'write as binary' mode
    with open(architecture_path,"wb") as file:
        
        # Write each bytestring to file
        file.write(total_num_layers_as_bytestring)
        file.write(layer_dimensions_as_bytestring)
        file.write(frequencies_as_bytestring)
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##

    return None

#==============================================================================
# Define a function to encode the weights/biases of each layer as a binary file
# containing strings of bytes

# Note: The np.method '.tobytes()' returns the same bytestring as 'struct.pack'

def EncodeParameters(network,original_centre,original_radius,parameters_path):
    
    # Extract a sorted list of the names of each layer in the network
    layer_names = network.get_weight_paths().keys()
    layer_names = sorted(list(layer_names),key=SortLayerNames)
    
    # Open the parameters file in 'write as binary' mode
    with open(parameters_path,"wb") as file:
    
        # Iterate through each of the network layers, in order 
        for layer_name in layer_names: 
            
            # Extract the layer weights and biases
            weights = network.get_weight_paths()[layer_name].numpy()
            
            # Flatten the layer weights and biases
            weights = np.ravel(weights,order="C").astype('float32')
        
            # Serialise weights into a string of bytes
            weights_as_bytestring = weights.tobytes(order="C")
                 
            # Write 'weight_as_bytestring' to file
            file.write(weights_as_bytestring)
            
        ##
         
        # Convert original centre to a numpy array
        original_centre = np.array(original_centre).astype('float32')
        
        # Serialise original centre into a string of bytes
        original_centre_as_bytestring = original_centre.tobytes(order="C")
        
        # Write 'original_centre_as_bytestring' to file
        file.write(original_centre_as_bytestring)
        
        # Convert original radius to a numpy array
        original_radius = np.array(original_radius).astype('float32')
        
        # Serialise original radius into a string of bytes
        original_radius_as_bytestring = original_radius.tobytes(order="C")
        
        # Write 'original_radius_as_bytestring' to file
        file.write(original_radius_as_bytestring)
        
        # Flush the buffer and close the file 
        file.flush()
        file.close()
    ##
    
    return None

#==============================================================================
# Define a function to sort the layer names alpha-numerically so that the saved
# weights are always in the correct order

def SortLayerNames(layer_name):
    
    layer_index = int(layer_name.split("_")[0][1:])

    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.50   
    
    return layer_index
    
#==============================================================================