""" Created: 29.01.2024  \\  Updated: 07.03.2024  \\   Author: Robert Sales """

#==============================================================================

import os, json, glob, trimesh, time, sys, datetime
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#==============================================================================
# Import user-defined libraries 

from SourceCode.data_management    import MakeMeshDataset,MakeGridDataset,LoadTrimesh
from SourceCode.compress_utilities import Logger
#==============================================================================

def PrecomputeMeshSDF(mesh_filepath,dataset_size,batch_size,sample_method,normalise,save_filepath):
        
    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",mesh_filepath.split("/")[-1]))
    mesh_obj_name = mesh_filepath.split("/")[-1].replace("_mesh.obj","")
    mesh,original_centre,original_radius = LoadTrimesh(mesh_filepath=mesh_filepath,normalise=normalise)
    
    # Check if the mesh is 'watertight' for computing SDFs
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight: SDFs can only be computed for watertight meshes. Please input watertight mesh.")
    ##    
       
    # Define mesh dataset path, make output path directory
    if not os.path.exists(save_filepath): os.makedirs(save_filepath) 
    save_mesh_data_path = os.path.join(save_filepath,"{:}_mesh_({:})_({:})_({:}).npy".format(mesh_obj_name,sample_method,dataset_size,"NORM" if normalise else "ORIG"))   
        
    # Make mesh dataset to supply coordinates and signed distances 
    if os.path.exists(save_mesh_data_path):
        print("\n{:30}'{}' {}".format("Mesh Dataset Already Exists:",save_mesh_data_path,"Skipping!"))
    else:
        print("\n{:30}'{}'".format("Precomputing Mesh Dataset:",save_mesh_data_path))
        tick = time.time()
        MakeMeshDataset(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size,save_filepath=save_mesh_data_path,show=False)        
        tock = time.time()
        print("\n{:30}{:.2f}".format("Elapsed Time:",(tock-tick)))
    ##
    
    return None

##

#==============================================================================

def PrecomputeGridSDF(mesh_filepath,grid_resolution,batch_size,bbox_scale,normalise,save_filepath):
        
    # Load the input mesh file using the 'trimesh' package    
    print("\n{:30}{}".format("Loaded Mesh:",mesh_filepath.split("/")[-1]))
    mesh_obj_name = mesh_filepath.split("/")[-1].replace("_mesh.obj","")
    mesh,original_centre,original_radius = LoadTrimesh(mesh_filepath=mesh_filepath,normalise=normalise)
    
    # Check if the mesh is 'watertight' for computing SDFs
    if mesh.is_watertight:
        print("\n{:30}{}".format("Edges:",mesh.vertices.shape[0]))
        print("\n{:30}{}".format("Faces:",mesh.faces.shape[0]   ))
    else:
        raise AssertionError("Input mesh not watertight: SDFs can only be computed for watertight meshes. Please input watertight mesh.")
    ##    
       
    # Define mesh dataset path, make output path directory
    if not os.path.exists(save_filepath): os.makedirs(save_filepath) 
    save_grid_data_path = os.path.join(save_filepath,"{:}_grid_({:})_({:})_({:}).npy".format(mesh_obj_name,grid_resolution,bbox_scale,"NORM" if normalise else "ORIG"))    
    
    # Make mesh dataset to supply coordinates and signed distances 
    if os.path.exists(save_grid_data_path):
        print("\n{:30}'{}' {}".format("Grid Dataset Already Exists:",save_grid_data_path,"Skipping!"))
    else:
        print("\n{:30}{}".format("Precomputing Grid Dataset:",save_grid_data_path))
        tick = time.time()
        MakeGridDataset(mesh=mesh,batch_size=batch_size,resolution=grid_resolution,bbox_scale=bbox_scale,save_filepath=save_grid_data_path,show=False)        
        tock = time.time()
        print("\n{:30}{:.2f}".format("Elapsed Time:",(tock-tick)))
    ##
    
    return None

##

#==============================================================================

def ViewPrecomputedMesh(mesh_filepath,dataset_size,batch_size,sample_method,normalise):

    mesh_obj_name = os.path.splitext(mesh_filepath.split("/")[-1])[0]
    load_filepath = os.path.join(os.path.dirname(mesh_filepath),"{:}_mesh_({:})_({:})_({:}).npy".format(mesh_obj_name,sample_method,dataset_size,"NORM" if normalise else "ORIG"))

    mesh,original_centre,original_radius = LoadTrimesh(mesh_filepath=mesh_filepath,normalise=normalise)

    sample_points_3d = np.load(load_filepath)[:,:-1]
    
    mesh.visual.face_colors = [255,0,0,128]
    
    points = trimesh.PointCloud(vertices=sample_points_3d,colors=[0,0,255,128])
    
    scene = trimesh.Scene([points,mesh])
    
    scene.show(flags={'wireframe':True},line_settings={'line_width':2.0,'point_size':0.5,})
    
    return None

##

#==============================================================================

def ViewPrecomputedGrid(mesh_filepath,grid_resolution,batch_size,bbox_scale,normalise):

    mesh_obj_name = os.path.splitext(mesh_filepath.split("/")[-1])[0]
    load_filepath = os.path.join(os.path.dirname(mesh_filepath),"{:}_grid_({:})_({:})_({:}).npy".format(mesh_obj_name,grid_resolution,bbox_scale,"NORM" if normalise else "ORIG"))

    mesh,original_centre,original_radius = LoadTrimesh(mesh_filepath=mesh_filepath,normalise=normalise)

    sample_points_3d = np.load(load_filepath)[:,:-1]
    
    mesh.visual.face_colors = [255,0,0,128]
    
    points = trimesh.PointCloud(vertices=sample_points_3d,colors=[0,0,255,128])
    
    scene = trimesh.Scene([points,mesh])
    
    scene.show(flags={'wireframe':True},line_settings={'line_width':2.0,'point_size':0.5,})
    
    return None

##

#==============================================================================    

if __name__ == "__main__":
          
    sys.stdout = Logger(logfile = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S.txt"))   
    
    ##
    
    config_filepaths = sorted(glob.glob("/Data/SDF_Compression_Datasets/*/*mesh_config.json"))
    
    config_filepaths = [x for x in config_filepaths if "mesh" in x]
    
    dataset_sizes = [1e5,1e6]
    
    sample_methods = ["surface","vertice","importance"]
    
    bbox_scales = [1.25]
    
    grid_resolutions = [32,64,128]
    
    ##
    
    for config_filepath in config_filepaths:
        
        for normalise in [True]:
        
            # DATA
            
            with open(config_filepath) as config_file: load_mesh_filepath = json.load(config_file)["mesh_filepath"]
            
            save_filepath = os.path.join("/",*load_mesh_filepath.split("/")[:-1])
            
            # MESH
            
            for dataset_size in dataset_sizes:
                
                for sample_method in sample_methods:
                               
                    PrecomputeMeshSDF(mesh_filepath=load_mesh_filepath,dataset_size=int(dataset_size),batch_size=1024,sample_method=sample_method,normalise=normalise,save_filepath=save_filepath)
                        
                    print("\n")
                    
                ##
                
            ##
            
            print("\n")
            
            # GRID
            
            for bbox_scale in bbox_scales:
                
                for grid_resolution in grid_resolutions:

                    PrecomputeGridSDF(mesh_filepath=load_mesh_filepath,grid_resolution=int(grid_resolution),batch_size=1024,bbox_scale=bbox_scale,normalise=normalise,save_filepath=save_filepath)

                    print("\n")
                    
                ##
                
            ##
        
            print("\n")
            
        ##
    
    ##
    
##        
    
else: pass

#==============================================================================
## Test 1
'''
mesh_filepath = "/Data/SDF_Compression_Datasets/turbostream_rotor/turbostream_rotor.obj"
dataset_size = int(1e6)
batch_size = int(1024)
sample_method = str("vertice")
normalise = True
ViewPrecomputedMesh(mesh_filepath,dataset_size,batch_size,sample_method,normalise)
'''
#==============================================================================
## Test 2
'''
mesh_filepath = "/Data/SDF_Compression_Datasets/turbostream_rotor/turbostream_rotor.obj"
grid_resolution = int(64)
batch_size = 1024
bbox_scale = float(1.1)
normalise = True
ViewPrecomputedGrid(mesh_filepath,grid_resolution,batch_size,bbox_scale,normalise)
'''
#==============================================================================