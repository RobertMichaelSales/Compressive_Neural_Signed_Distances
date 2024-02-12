""" Created: 29.01.2024  \\  Updated: 07.02.2024  \\   Author: Robert Sales """

#==============================================================================

import os, time, json, psutil, sys, gc, datetime, glob, trimesh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.enable()

import numpy as np

#==============================================================================
# Import user-defined libraries 

from SourceCode.data_management import MakeMeshDataset,MakeGridDataset

#==============================================================================

def Precompute_Mesh_SDF(mesh_filepath,dataset_size,batch_size,sample_method,save_filepath):
        
    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",mesh_filepath.split("/")[-1]))
    mesh_obj_name = os.path.splitext(mesh_filepath.split("/")[-1])[0]
    mesh = trimesh.load(mesh_filepath)
    
    # Check if the mesh is 'watertight' for computing SDFs
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight: SDFs can only be computed for watertight meshes. Please input watertight mesh.")
    ##    
       
    # Define mesh dataset path, make output path directory
    if not os.path.exists(save_filepath): os.makedirs(save_filepath) 
    save_mesh_data_path = os.path.join(save_filepath,"{:}_mesh_[{:}]_[{:}].npy".format(mesh_obj_name,sample_method,dataset_size))   
        
    # Make mesh dataset to supply coordinates and signed distances 
    if os.path.exists(save_mesh_data_path):
        print("\n{:30}{}".format("Mesh Dataset Already Exists:","Skipping..."))
    else:
        print("\n{:30}{}".format("Precomputing Mesh Dataset:",save_mesh_data_path))
        MakeMeshDataset(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size,save_filepath=save_mesh_data_path,show=False)        
    ##
    
    return None

##

#==============================================================================

def Precompute_Grid_SDF(mesh_filepath,grid_resolution,batch_size,bbox_scale,save_filepath):
        
    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",mesh_filepath.split("/")[-1]))
    mesh_obj_name = os.path.splitext(mesh_filepath.split("/")[-1])[0]
    mesh = trimesh.load(mesh_filepath)
    
    # Check if the mesh is 'watertight' for computing SDFs
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight: SDFs can only be computed for watertight meshes. Please input watertight mesh.")
    ##    
       
    # Define mesh dataset path, make output path directory
    if not os.path.exists(save_filepath): os.makedirs(save_filepath) 
    save_grid_data_path = os.path.join(save_filepath,"{:}_grid_[{:}]_[{:}].npy".format(mesh_obj_name,grid_resolution,bbox_scale))
    
    # Make mesh dataset to supply coordinates and signed distances 
    if os.path.exists(save_grid_data_path):
        print("\n{:30}{}".format("Mesh Dataset Already Exists:","Skipping..."))
    else:
        print("\n{:30}{}".format("Precomputing Mesh Dataset:",save_grid_data_path))
        MakeGridDataset(mesh=mesh,batch_size=batch_size,resolution=grid_resolution,bbox_scale=bbox_scale,save_filepath=save_grid_data_path,show=False)        
    ##
    
    return None

##

#==============================================================================    

if __name__ == "__main__":
        
    ##
    
    config_filepaths = glob.glob("/Data/SDF_Compression_Datasets/armadillo/armadillo_config.json")
            
    dataset_sizes = [1e6]
    
    sample_methods = ["vertice"]
    
    bbox_scales = [1.1]
    
    grid_resolutions = [64]
    
    ##
    
    for config_filepath in config_filepaths:
        
        # DATA
        
        with open(config_filepath) as config_file: load_mesh_filepath = json.load(config_file)["mesh_filepath"]
        
        save_filepath = os.path.join("/",*load_mesh_filepath.split("/")[:-1])
        
        # MESH
        
        for dataset_size in dataset_sizes:
            
            for sample_method in sample_methods:
            
                Precompute_Mesh_SDF(mesh_filepath=load_mesh_filepath,dataset_size=int(dataset_size),batch_size=1024,sample_method=sample_method,save_filepath=save_filepath)
                
            ##
            
        ##
        
        print("\n")
        
        # GRID
        
        for bbox_scale in bbox_scales:
            
            for grid_resolution in grid_resolutions:
                
                Precompute_Grid_SDF(mesh_filepath=load_mesh_filepath,grid_resolution=int(grid_resolution),batch_size=1024,bbox_scale=bbox_scale,save_filepath=save_filepath)
                
            ##
            
        ##
        
        print("\n")
    
    ##
    
##        
    
else: pass

#==============================================================================