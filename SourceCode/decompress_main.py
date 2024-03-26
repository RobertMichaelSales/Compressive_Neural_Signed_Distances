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

layer_dimensions,frequencies = DecodeArchitecture(architecture_path=architecture_path)
  
SquashSDF = ConstructNetwork(layer_dimensions=layer_dimensions,frequencies=frequencies) 

parameters,original_centre,original_radius = DecodeParameters(network=SquashSDF,parameters_path=parameters_path)

AssignParameters(network=SquashSDF_2,parameters=parameters)  
