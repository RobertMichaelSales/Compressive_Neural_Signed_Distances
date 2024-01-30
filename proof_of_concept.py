""" Created: 22.01.2024  \\  Updated: 22.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import os
import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt

#==============================================================================
# Import user-defined libraries 

#==============================================================================

mesh_filename = "/home/rms221/Documents/Compressive_Signed_Distance_Functions/ICML2021/neuralImplicitTools/data/bumpy-cube.obj"

mesh = trimesh.load(mesh_filename)
sphere = mesh.bounding_sphere

# SAMPLE_SPHERE_RADIUS = (sphere.bounds.ptp()/2)
# sample_points = ((np.random.rand(100000,3)-0.5)*2)
# sample_points = SAMPLE_SPHERE_RADIUS * sample_points[np.linalg.norm(sample_points, axis=1) <= 1.5]
# ptcd = trimesh.PointCloud(sample_points)

trimesh.Scene([mesh,sphere,ptcd]).show(flags={'wireframe': True,},line_settings={'line_width':1,'point_size':1})


# ptcd = trimesh.PointCloud(mesh.vertices)
# trimesh.Scene(ptcd).show(line_settings={'point_size':1})

# is_watertight = mesh.is_watertight

# mesh_f = mesh.faces
# mesh_v = mesh.vertices

# sphere_centre = mesh.bounding_sphere.center


# bbox = mesh.bounding_box

# bbox_cx,bbox_cy,bbox_cz = bbox.centroid

# bbox_lx,bbox_ly,bbox_lz = bbox.extents

# scale = 1.0

# grid_xs = np.linspace((bbox_cx-(bbox_lx*scale/2)),(bbox_cx+(bbox_lx*scale/2)),32)
# grid_ys = np.linspace((bbox_cy-(bbox_ly*scale/2)),(bbox_cy+(bbox_ly*scale/2)),32)
# grid_zs = np.linspace((bbox_cz-(bbox_lz*scale/2)),(bbox_cz+(bbox_lz*scale/2)),32)

# grid_3d = np.stack(np.meshgrid(grid_xs,grid_ys,grid_zs,indexing="ij"),axis=-1)

# grid = grid_3d.reshape((-1,3))

# import time
# tick = time.time()
# sdf = trimesh.proximity.signed_distance(mesh,grid)
# tock = time.time()
# print(tock-tick)

# sdf = sdf.reshape((32,32,32,1))

# for i in range(30):

#     fig = plt.figure(figsize=(4,4),constrained_layout=True)
#     gridspec = fig.add_gridspec(nrows=1,ncols=1,height_ratios=[1],width_ratios=[1])
#     ax1 = fig.add_subplot(gridspec[0,0])    
#     ax1.contourf(grid_3d[:,:,i,0],grid_3d[:,:,i,1],sdf[:,:,i,0],levels=32)
#     ax1.contour(grid_3d[:,:,i,0],grid_3d[:,:,i,1],sdf[:,:,i,0],levels=[0.0],color="black")
#     plt.show()
