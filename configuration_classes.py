""" Created: 29.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

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