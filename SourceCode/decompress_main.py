""" Created: 11.03.2024  \\  Updated: 15.03.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

#==============================================================================
# Import user-defined libraries 

from network_decoder         import DecodeParameters,DecodeArchitecture,AssignParameters
from network_model           import ConstructNetworkBASIC

#==============================================================================

architecture_path = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/AuxFiles/outputs/squashsdf_savesdf/architecture.bin"

parameters_path = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/AuxFiles/outputs/squashsdf_savesdf/parameters.bin"


##

layer_dimensions,frequencies,activation = DecodeArchitectureBASIC(architecture_path=architecture_path)
  
SquashSDF = ConstructNetworkBASIC(layer_dimensions=layer_dimensions,frequencies=frequencies,activation=activation) 

##

layer_dimensions,frequencies = DecodeArchitectureSIREN(architecture_path=architecture_path)

SquashSDF = ConstructNetworkSIREN(layer_dimensions=layer_dimensions,frequencies=frequencies)    

##

layer_dimensions,activation,gaussian_kernel = DecodeArchitectureGAUSS(architecture_path=architecture_path)

SquashSDF = ConstructNetworkSIREN(layer_dimensions=layer_dimensions,frequencies=None,stddev=None,activation=activation,gaussian_kernel=gaussian_kernel)    

##

parameters,original_centre,original_radius = DecodeParameters(network=SquashSDF_2,parameters_path=parameters_path)

AssignParameters(network=SquashSDF_2,parameters=parameters)  
