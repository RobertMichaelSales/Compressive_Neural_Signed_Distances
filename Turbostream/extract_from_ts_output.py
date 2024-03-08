""" Created: 22.02.2024  \\  Updated: 06.03.2024  \\   Author: Robert Sales """

# Note: Run this file from the command line using "pvpython filename.py"

#==============================================================================
# trace generated using paraview version 5.9.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 9

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#### reset paraview session
ResetSession()

#### import any other modules
import numpy as np
import os,json,glob,vtk

from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter
    
#==============================================================================
# A recursive function for producing a dictionary with block indices for multi-
# block Turbostream datasets - allowing easier specification in "ExtractBlocks"

# E.g. indices["Domains"]["block_index"] = 0 (gives both Domain 0 and Domain 1)
# E.g. indices["Domains"]["Domain 0"]["block_index"] = 1 (giving only Domain 0)

def GetBlockIndices(compositeDataInformation,index=1):
    
    # Create empty dictionary of local indices
    indices = {}
        
    # Check if current parent has any children
    if (compositeDataInformation.GetNumberOfChildren() != 0):
        
        # Iterate through children blocks and indices
        for i in range(compositeDataInformation.GetNumberOfChildren()):
            
            # Append current "BlockIndex" and any children
            indices[compositeDataInformation.GetName(i)] = {"BlockIndex":index}
            index = index + 1
            local_indices,index = GetBlockIndices(compositeDataInformation.GetDataInformation(i).GetCompositeDataInformation(),index)
            indices[compositeDataInformation.GetName(i)].update(local_indices) 
            
        ##  
        
    ##
    
    return indices,index

##
    
#==============================================================================
# A function for returning numpy arrays of input coordinates (volume = x, y, z)
# and output scalars (values = mach) for an input dataset. The volume and value 
# array names must be in 'point_array_names' else returns a (None,None) tuple.

def GetPointArraysAsNumpy(dataset,point_array_names,volume_array_names,values_array_names):
  
    import numpy as np
    
    # Check if entries in 'volume_array_names' are valid, then stack 4-D tensor
    if not (all([x in point_array_names for x in volume_array_names])):
        raise NameError("Invalid volume_array_names '{}' in point_array_names '{}'".format(volume_array_names,point_array_names))
    else:
        volume = np.stack([vtk_to_numpy(dataset.GetPointData().GetArray(array_name)) for array_name in volume_array_names],axis=-1)    
    ##

    # Check if entries in 'values_array_names' are valid, then stack 4-D tensor
    if not (all([x in point_array_names for x in values_array_names])):
        raise NameError("Invalid values_array_names '{}' in point_array_names '{}'".format(values_array_names,point_array_names))
    else:
        values = np.stack([vtk_to_numpy(dataset.GetPointData().GetArray(array_name)) for array_name in values_array_names],axis=-1)    
    ##
            
    # Convert to float32 precision    
    return volume.astype("float32"),values.astype("float32")

##

#==============================================================================
# A function that extracts (1): the domain geometry as a '.obj' triangular mesh 
# for learning neural SDFs and (2): the input coordinate and scalar output pair
# for learning neural volumetric scalar fields. 

# Example input argumements to Export() are:

# E.g. topology_filepath = '/home/rms221/turbostream/rotor67_tutorial_release/runs/AR_0.900/input_1.hdf5'
# E.g. value_filepath = '/home/rms221/turbostream/rotor67_tutorial_release/runs/AR_0.900/output_1.hdf5'
# E.g. output_filepath = '/home/rms221/turbostream/extracts'
    
# E.g. domain_name = "Domain 0"

# E.g. volume_array_names = ["x","y","z"]
# E.g. values_array_names = ["mach"]

# E.g. gamma=1.4
# E.g. cp=1005
# E.g. tref=288.15
# E.g. pref=101325


def Export(topology_filepath,value_filepath,output_filepath,domain_name,volume_array_names,values_array_names,Gamma,Cp,Tref,Pref):    
    
    #==========================================================================
    
    # Create a new 'TS Turbostream Reader'
    tSTurbostreamReader1 = TSTurbostreamReader(registrationName='TSTurbostreamReader1',Topologyfilepath='',Valuefilepath='')
            
    # Check filepath exists, properties modified on tSTurbostreamReader1
    if not os.path.isfile(topology_filepath):
        raise FileNotFoundError("File '{}' not found.".format(topology_filepath))
    else:
        tSTurbostreamReader1.Topologyfilepath = topology_filepath
    ##
    
    # Check filepath exists, properties modified on tSTurbostreamReader1
    if not os.path.isfile(value_filepath   ):
        raise FileNotFoundError("File '{}' not found.".format(value_filepath))
    else:
        tSTurbostreamReader1.Valuefilepath = value_filepath    
    ## 
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=tSTurbostreamReader1)
    
    #==========================================================================
    
    # Create a new 'TS Secondary Variables'
    tSSecondaryVariables1 = TSSecondaryVariables(registrationName='TSSecondaryVariables1',Input=tSTurbostreamReader1)
    
    # Properties modified on tSSecondaryVariables1
    tSSecondaryVariables1.Gamma = Gamma
    tSSecondaryVariables1.Cp = Cp
    tSSecondaryVariables1.Tref = Tref
    tSSecondaryVariables1.Pref = Pref
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=tSSecondaryVariables1) 
    
    #==========================================================================
    
    # Get block indices 
    block_indices,num_blocks = GetBlockIndices(compositeDataInformation=tSTurbostreamReader1.GetDataInformation().GetCompositeDataInformation())
    
    domain_names = [x for x in block_indices["Domains"].keys() if x != "BlockIndex"]
    
    if not domain_name in domain_names: raise NameError("Domain name '{}' not in '{}'".format(domain_name,domain_names))
        
    # Create a new 'Extract Block'
    extractBlock1 = ExtractBlock(registrationName='ExtractBlock1',Input=tSSecondaryVariables1)
    
    # Properties modified on extractBlock1
    extractBlock1.BlockIndices = [block_indices["Domains"][domain_name]["BlockIndex"]]
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=extractBlock1)
    
    #==========================================================================
    
    # create a new 'Merge Blocks'
    mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1',Input=extractBlock1)
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=mergeBlocks1)
    
    #==========================================================================
    
    # Create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1',Input=mergeBlocks1)
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=extractSurface1)
    
    # Create a new 'Triangulate'
    triangulate1 = Triangulate(registrationName='Triangulate1',Input=extractSurface1)
    
    # Update pipeline
    UpdatePipeline(time=0.0,proxy=triangulate1)
    
    #==========================================================================
    
    # Create output folder
    if not os.path.exists(output_filepath): os.makedirs(output_filepath)
    
    #==========================================================================
            
    # Make mesh output filepath
    mesh_filepath = os.path.join(output_filepath,"{}_mesh.obj".format(domain_name.lower().replace(" ","_")))
    
    # Save data
    SaveData(filename=mesh_filepath,proxy=triangulate1)
    
    # Make mesh config filepath
    mesh_config_filepath = mesh_filepath.replace(".obj","_config.json")
    
    # Get number of points
    number_of_points = mergeBlocks1.GetDataInformation().DataInformation.GetNumberOfPoints()
    
    # Make mesh config dict
    mesh_config = {"mesh_filepath":mesh_filepath,"original_volume_size":number_of_points}
    
    # Make mesh config file
    with open(mesh_config_filepath,"w") as mesh_config_file: json.dump(mesh_config,mesh_config_file,indent=4,sort_keys=True)
    
    #==========================================================================
    
    # Get number of point arrays
    numberOfPointArrays = mergeBlocks1.GetPointDataInformation().GetNumberOfArrays()
    
    # Make proxy dataset
    dataset = servermanager.Fetch(mergeBlocks1)
    
    # Get point array names
    point_array_names = [dataset.GetPointData().GetArray(i).GetName() for i in range(numberOfPointArrays)]
    
    # Extract volume and values
    volume,values = GetPointArraysAsNumpy(dataset=dataset,point_array_names=point_array_names,volume_array_names=volume_array_names,values_array_names=values_array_names)
    
    # Combine volume and values
    data = np.concatenate((volume,values),axis=-1)
    
    # Make data output filepath
    data_filepath = os.path.join(output_filepath,"{}_{}.npy".format(domain_name.lower().replace(" ","_"),"_".join(values_array_names)))
    
    # Save as Numpy file 
    np.save(file=data_filepath,arr=data)
    
    # Make mesh config dict
    i_cols = [x for x in range(data.shape[-1])][:volume.shape[-1]]
    o_cols = [x for x in range(data.shape[-1])][values.shape[-1]:]
    data_config = {"columns":[i_cols,o_cols,[]],"dtype":"float32","i_filepath":data_filepath,"normalise":True,"shape":list(data.shape),"tabular":True}       
    
    # Make data config filepath
    data_config_filepath = data_filepath.replace(".npy","_config.json")
    
    # Make mesh config file
    with open(data_config_filepath,"w") as data_config_file: json.dump(data_config,data_config_file,indent=4,sort_keys=True)
    
    #==========================================================================
    
    return None
    
##    
    
#==============================================================================   
# This code will execute in the event that this file is called via the terminal

# E.g. "python extract_from_ts_output.py"

if __name__ == "__main__":
  
    topology_filepath = '/home/rms221/turbostream/rotor67_tutorial_release/runs/AR_0.900/input_1.hdf5'
    value_filepath = '/home/rms221/turbostream/rotor67_tutorial_release/runs/AR_0.900/output_1.hdf5'
    output_filepath = '/home/rms221/turbostream/extracts'
    
    volume_array_names = ["x","y","z"]
    values_array_names = ["mach"]
    
    Gamma=1.4
    Cp=1005
    Tref=288.15
    Pref=101325
    
    domain_name = "Domain 0"
    
    Export(topology_filepath=topology_filepath,value_filepath=value_filepath,output_filepath=output_filepath,domain_name=domain_name,volume_array_names=volume_array_names,values_array_names=values_array_names,Gamma=Gamma,Cp=Cp,Tref=Tref,Pref=Pref)
    
    domain_name = "Domain 1"
    
    Export(topology_filepath=topology_filepath,value_filepath=value_filepath,output_filepath=output_filepath,domain_name=domain_name,volume_array_names=volume_array_names,values_array_names=values_array_names,Gamma=Gamma,Cp=Cp,Tref=Tref,Pref=Pref)
    
else: pass
#==============================================================================