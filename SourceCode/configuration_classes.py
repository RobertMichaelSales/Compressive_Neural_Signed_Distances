""" Created: 29.01.2024  \\  Updated: 07.02.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import numpy as np

#==============================================================================
# Define a class with dictionary functionality to store arbitrary attributes 

class GenericConfigurationClass(dict):
    
    #==========================================================================
    # Define the initialisation constructor function to convert a dictionary to
    # an object: dictionary[key] = value --> object.key = value

    def __init__(self,config_dict={}):
        
        for key in config_dict.keys():
            
            self.__setitem__(key,config_dict[key])
    
        return None
    
    ##
    
    #==========================================================================
    # Redefine the method which handles/intercepts inexistent attribute lookups
    
    def __getattr__(self, key):
        
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__,key))
        
        return None
        
    ##
    
    #==========================================================================    
    # Redefine the method that is called when attribute assignment is attempted
    # on the class
    
    def __setattr__(self,key,value):
        
        self.__setitem__(key,value)
        
        return None
    
    ##
    
    #==========================================================================
    # Redefine the method that is called when implementing assignment self[key]
    # such that the class has dictionary functionality
    
    def __setitem__(self,key,value):
        
        super(GenericConfigurationClass,self).__setitem__(key,value)
        self.__dict__.update({key:value})
    
        return None
    
    ##
    
    #==========================================================================
    
##    
    
#==============================================================================
# Define a class to manage the network configuration

class NetworkConfigurationClass(GenericConfigurationClass):
    
    #==========================================================================
    # Define the initialisation constructor function (with inheritance from the
    # 'GenericConfigurationClass' class)

    def __init__(self,config_dict={}):
        
        super().__init__(config_dict)
        
        return None
    
    ##
    
    #==========================================================================
    # Define a function to generate the network structure/dimensions from input
    # dimensions, output dimensions and input size.
    
    def GenerateStructure(self,i_dimensions,o_dimensions,original_volume_size):
            
        # Assert that self.network_architecture is one of the accepted options
        if (self.network_architecture.upper() not in ["BASIC","SIREN"]):
            raise AssertionError("Network arcitecture 'network_config.network_architecture' must be in '[BASIC, SIREN]'")
        ##
        
        # Extract the useful internal parameters from the 'input_data' object
        self.i_dimensions = i_dimensions
        self.o_dimensions = o_dimensions
        self.original_volume_size = original_volume_size
        
        # Compute the network's target capacity
        self.target_capacity = int((self.original_volume_size*self.i_dimensions)/self.target_compression_ratio)
        
        # If BASIC then calculate GetNetworkCapacityBASIC accordingly
        if (self.network_architecture.upper() == "BASIC"):
            
            # Compute the widths of hidden layers
            self.neurons_per_layer = self.GetNeuronsPerLayerBASIC() 
            
            # Compute the network's total capacity
            self.actual_capacity = self.GetNetworkCapacityBASIC()
            
        ##
        
        # If SIREN then calculate GetNetworkCapacitySIREN accordingly
        if (self.network_architecture.upper() == "SIREN"):
            
            # Compute the widths of hidden layers
            self.neurons_per_layer = self.GetNeuronsPerLayerSIREN() 
            
            # Compute the network's total capacity
            self.actual_capacity = self.GetNetworkCapacitySIREN()
            
        ##

        # Compute the network's actual compression ratio
        self.actual_compression_ratio = float((self.original_volume_size*self.i_dimensions)/self.actual_capacity)
        
        # Write the network architecture as a list of layer dimensions
        self.layer_dimensions = [self.i_dimensions] + ([self.neurons_per_layer]*self.hidden_layers) + [self.o_dimensions]
        
        # Print the network architecture: SIREN, BASIC
        print("\n{:30}'{}'".format("Network Architecture:",self.network_architecture))
        
        # Print the network's target compression ratio
        print("\n{:30}{:.2f}".format("Target compression ratio:",self.target_compression_ratio))
        
        # Print the network's actual compression ratio
        print("\n{:30}{:.2f}".format("Actual compression ratio:",self.actual_compression_ratio))
        
        # Print the network's encoding Frequencies
        print("\n{:30}{}".format("Encoding frequencies:",self.frequencies))
        
        # Print the network's layer dimensions
        print("\n{:30}{}".format("Network dimensions:",self.layer_dimensions))

        return None
    
    ##

    #==========================================================================
    # Define a function to compute the minimum number of neurons needed by each 
    # layer in order to achieve (just exceed) the target compression ratio

    def GetNeuronsPerLayerBASIC(self):
      
        # Start searching from the minimum of 1 neuron per layer
        self.neurons_per_layer = int(self.minimum_neurons_per_layer)
                
        # Incriment neurons until the network capacity exceeds the target size
        while (self.GetNetworkCapacityBASIC() < self.target_capacity):
            
            self.neurons_per_layer = self.neurons_per_layer + 1
            
        ##
          
        # Determine the first neuron count that exceeds the target compression
        self.neurons_per_layer = self.neurons_per_layer - 1
        
        return self.neurons_per_layer
    
    ##
    
    #==========================================================================
    # Define a function to calculate the total number of network parameters for
    # a basic network architecture (i.e. layer dimensions/neurons)
    
    # The network structure can be summarised as follows:
    # [input_layer      -> hidden_layer] + 
    # [hidden_layer     -> hidden_layer] +
    # [hidden_layer     -> output_layer]  
    
    def GetNetworkCapacityBASIC(self):    
        
        # Determine the number of inter-layer operations (i.e. total layers)
        self.total_layers = self.hidden_layers + 1     
                  
        # Set the total number of parameters to zero
        self.actual_capacity = 0                                                         
          
        #Iterate through each layer in the network (including input/output)
        for layer in np.arange(self.total_layers):
          
            # [input_layer -> hidden_layer]
            if (layer==0):                             
                
                # Determine the input and output dimensions of each layer
                if (self.frequencies > 0):
                    i_dimensions = self.i_dimensions * self.frequencies * 2
                else:
                    i_dimensions = self.i_dimensions
                ##
                      
                o_dimensions = self.neurons_per_layer
                
                # Add parameters from the weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                  
            ##                
          
            # [hidden_layer -> output_layer]
            elif (layer==self.total_layers-1):     
    
                # Determine the input and output dimensions of each layer
                i_dimensions = self.neurons_per_layer
                o_dimensions = self.o_dimensions
                
                # Add parameters from the weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                  
            ##
          
            # [hidden_layer -> hidden_layer]
            else:                         
                
                # Determine the input and output dimensions of each layer    
                i_dimensions = self.neurons_per_layer                          
                o_dimensions = self.neurons_per_layer
                
                # Add parameters from the weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                
            ##
            
        ##
                  
        return self.actual_capacity
    
    ##
    
    #==========================================================================
    # Define a function to compute the minimum number of neurons needed by each 
    # layer in order to achieve (just exceed) the target compression ratio

    def GetNeuronsPerLayerSIREN(self):
      
        # Start searching from the minimum of 1 neuron per layer
        self.neurons_per_layer = int(self.minimum_neurons_per_layer)
                
        # Incriment neurons until the network capacity exceeds the target size
        while (self.GetNetworkCapacitySIREN() < self.target_capacity):
            
            self.neurons_per_layer = self.neurons_per_layer + 1
            
        ##
          
        # Determine the first neuron count that exceeds the target compression
        self.neurons_per_layer = self.neurons_per_layer - 1
        
        return self.neurons_per_layer
    
    ##
    
    #==========================================================================
    # Define a function to calculate the total number of network parameters for
    # a siren network architecture (i.e. layer dimensions/neurons)
    
    # Note - total_layers is '+2' because the first sine layer, which is needed
    # to make sure that the residual tensors are of the same shape, and because 
    # the total_layers actually means "total blocks" in reality.
    
    # The network structure can be summarised as follows:
    # [input_layer      -> sine_layer] + 
    # [sine_layer/block -> sine_block] +
    # [sine_block       -> output_layer]  
    
    def GetNetworkCapacitySIREN(self):    
        
        # Determine the number of inter-layer operations (i.e. total layers)
        self.total_layers = self.hidden_layers + 2     
                  
        # Set the total number of parameters to zero
        self.actual_capacity = 0                                                         
          
        #Iterate through each layer in the network (including input/output)
        for layer in np.arange(self.total_layers):
          
            # [input_layer -> hidden_layer]
            if (layer==0):                             
                
                # Determine the input and output dimensions of each layer
                if (self.frequencies > 0):
                    i_dimensions = self.i_dimensions * self.frequencies * 2
                else:
                    i_dimensions = self.i_dimensions
                ##
                      
                o_dimensions = self.neurons_per_layer
                
                # Add parameters from the weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                  
            ##                
          
            # [hidden_layer -> output_layer]
            elif (layer==self.total_layers-1):     
    
                # Determine the input and output dimensions of each layer
                i_dimensions = self.neurons_per_layer
                o_dimensions = self.o_dimensions
                
                # Add parameters from the weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                  
            ##
          
            # [hidden_layer -> hidden_layer]
            else:                         
                
                # Determine the input and output dimensions of each layer    
                i_dimensions = self.neurons_per_layer                          
                o_dimensions = self.neurons_per_layer
                
                # Add parameters from 1st weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                
                # Add parameters from 2nd weight matrix and bias vector
                self.actual_capacity += (i_dimensions * o_dimensions) + o_dimensions
                
            ##
            
        ##
                  
        return self.actual_capacity
    
    ##
    
##   
    
#==============================================================================