""" Created: 23.01.2024  \\  Updated: 29.01.2024  \\   Author: Robert Sales """

#==============================================================================
# Import libraries and set flags

import trimesh, time
import numpy as np
import tensorflow as tf

#==============================================================================

class MeshDataset():
    
    def __init__(self,mesh,batch_size,sample_method,dataset_size):
        
        self.mesh = mesh
        
        self.batch_size = batch_size
        
        self.sample_method = sample_method
        
        self.dataset_size = dataset_size 

        self.verts = mesh.vertices
        
        self.faces = mesh.faces
        
        self.stdev = 0.10
        
        self.ratio = 10.0
        
        self.scale = 60.0
        
        self.GetFaceAreaCDF()
        
    ##
    
    #==============================================================================
    
    # Returns a cumulative distribution function of face areas (with min = 0, max = 1) in the same order as 'faces'
    
    def GetFaceAreaCDF(self):
                
        vector_ab = self.verts[self.faces[:,0],:] - self.verts[self.faces[:,1],:]
        
        vector_ac = self.verts[self.faces[:,0],:] - self.verts[self.faces[:,2],:]
        
        face_areas_abc = np.linalg.norm(np.cross(vector_ab,vector_ac),axis=1)
        
        face_areas_abc = face_areas_abc/ np.sum(face_areas_abc)
        
        self.face_areas_cdf = np.concatenate(([0],np.cumsum(face_areas_abc)))
    
    ##    

    #==============================================================================
    
    # Returns coordinates and signed distances of points randomly distributed within the bounding sphere of a mesh
    
    def UniformSampler(self,n_samples):
        
        bounding_sphere_radius = (1.0)*(np.ptp(self.mesh.bounding_sphere.bounds)/2.0)
        
        sample_points_3d = np.empty(shape=(0,3))
        
        remaining_points = n_samples
    
        while (remaining_points > 0):
            
            random_points = np.random.uniform(low=-bounding_sphere_radius,high=+bounding_sphere_radius,size=(remaining_points,3))
    
            random_points = random_points[np.linalg.norm(random_points,axis=1)<=bounding_sphere_radius]
            
            sample_points_3d = np.concatenate([sample_points_3d,random_points])
    
            remaining_points = n_samples - sample_points_3d.shape[0]
            
        ##
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
        
        return sample_points_3d, signed_distances
    
    ##    
    
    #==========================================================================
    
    # Returns coordinates and signed distances of points randomly sampled (with normally distributed offset) from the surface of a mesh
    # Barycentric coordinate system for sampling points from a triangle's vertices https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    
    def SurfaceSampler(self,n_samples):
    
        sample_face_indices = np.array(np.searchsorted(a=self.face_areas_cdf,v=np.random.rand(n_samples),side="right")-1)
    
        vertex_a = self.verts[self.faces[sample_face_indices,0],:]
        
        vertex_b = self.verts[self.faces[sample_face_indices,1],:]
        
        vertex_c = self.verts[self.faces[sample_face_indices,2],:]
        
        lambdas = np.random.rand(n_samples,3)
        
        lambdas = np.expand_dims((lambdas / np.expand_dims(np.sum(lambdas,axis=1),axis=-1)), axis=-1)
                
        sample_points_3d = (lambdas[:,0] * vertex_a) + (lambdas[:,1] * vertex_b) + (lambdas[:,2] * vertex_c)
        
        sample_points_3d = np.random.normal(loc=sample_points_3d,scale=self.stdev)
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
        
        return sample_points_3d, signed_distances
        
    ##
        
    #==========================================================================
    
    # Returns coordinates and signed distances of points randomly sampled (with normally distributed offset) from the nodes of a mesh
    
    def VerticeSampler(self,n_samples):
        
        sample_points_3d = self.verts[np.random.choice(self.verts.shape[0],n_samples),:]
    
        sample_points_3d = np.random.normal(loc=sample_points_3d,scale=self.stdev)
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
    
        return sample_points_3d, signed_distances
    
    ##
    
    #==========================================================================
    
    # Returns coordinates and signed distances of points sampled using Davies et al.'s importance sampling strategy, for any given mesh
    # "On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes" -> https://arxiv.org/abs/2009.09808
    
    def ImportanceSampler(self,n_samples):
        
        uniform_points = self.UniformSampler(n_samples=(n_samples*self.ratio))
        
        signed_distances = trimesh.proximity.signed_distance(self.mesh,uniform_points)

        importance = np.exp(((-self.scale) * np.abs(signed_distances)))
                
        importance_cdf = np.concatenate(([0],np.cumsum(importance / np.sum(importance))))
        
        sample_point_indices = np.array(np.searchsorted(a=importance_cdf,v=np.random.rand(n_samples),side="right")-1)
        
        sample_points_3d = uniform_points[sample_point_indices,:]
        
        signed_distances = np.expand_dims(a=signed_distances[sample_point_indices,:],axis=-1)
        
        return sample_points_3d, signed_distances
    
    ##
    
    #==========================================================================
    
    def GenerateData(self):
        
        self.sample_points_3d = np.empty(shape=(0,3))
        
        self.signed_distances = np.empty(shape=(0,1))
        
        while self.signed_distances.shape[0] < self.dataset_size:
            
            if self.sample_method == "uniform": 
                sample_points_3d_batch,signed_distances_batch = self.UniformSampler(self.batch_size)
            ##
            
            if self.sample_method == "surface": 
                sample_points_3d_batch,signed_distances_batch = self.SurfaceSampler(self.batch_size)
            ##
            
            if self.sample_method == "vertice": 
                sample_points_3d_batch,signed_distances_batch = self.VerticeSampler(self.batch_size)
            ##
            
            if self.sample_method == "importance": 
                sample_points_3d_batch,signed_distances_batch = self.ImportanceSampler(self.batch_size)
            ##
        
            self.sample_points_3d = np.concatenate([self.sample_points_3d,sample_points_3d_batch])
            
            self.signed_distances = np.concatenate([self.signed_distances,signed_distances_batch])
            
            print(ProgressBar(current=self.sample_points_3d.shape[0],end=self.dataset_size),end="")
        
        ##
        
        self.sample_points_3d = self.sample_points_3d[:self.dataset_size,:]
        
        self.signed_distances = self.signed_distances[:self.dataset_size,:]
                        
        return None
    
    ##
    
#==============================================================================

def MakeMeshDataset(mesh,batch_size,sample_method,dataset_size,save_filepath,show=False):
    
    mesh_data = MeshDataset(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size)
    
    mesh_data.GenerateData()
    
    np.save(file=save_filepath,arr=np.concatenate((mesh_data.sample_points_3d,mesh_data.signed_distances),axis=-1))
    
    sample_points_3d = tf.convert_to_tensor(mesh_data.sample_points_3d.astype("float32"))
    
    signed_distances = tf.convert_to_tensor(mesh_data.signed_distances.astype("float32"))

    dataset = tf.data.Dataset.from_tensor_slices((sample_points_3d,signed_distances))
    
    dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=dataset_size,reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset.dataset_size = dataset_size
    dataset.size = len(dataset)
    dataset.batch_size = batch_size
    
    if show:
        mesh_data.mesh.visual.face_colors = [255,0,0,128]
        points = trimesh.PointCloud(vertices=mesh_data.sample_points_3d,colors=[0,0,255,128])
        trimesh.Scene([points,mesh_data.mesh]).show(line_settings={'point_size':0.5,})
    ##
    
    return dataset

##

#==============================================================================

def LoadMeshDataset(mesh,batch_size,sample_method,dataset_size,load_filepath,show=False):
    
    mesh_data = MeshDataset(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size)
    
    mesh_data.sample_points_3d = np.load(load_filepath)[:,:-1]
    
    mesh_data.signed_distances = np.load(load_filepath)[:,-1:] 
    
    sample_points_3d = tf.convert_to_tensor(mesh_data.sample_points_3d.astype("float32"))
    
    signed_distances = tf.convert_to_tensor(mesh_data.signed_distances.astype("float32"))

    dataset = tf.data.Dataset.from_tensor_slices((sample_points_3d,signed_distances))
    
    dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=dataset_size,reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset.dataset_size = dataset_size
    dataset.size = len(dataset)
    dataset.batch_size = batch_size
    
    if show:
        mesh_data.mesh.visual.face_colors = [255,0,0,128]
        points = trimesh.PointCloud(vertices=mesh_data.sample_points_3d,colors=[0,0,255,128])
        trimesh.Scene([points,mesh_data.mesh]).show(flags={'wireframe':True},line_settings={'line_width':2.0,'point_size':0.5,})
    ##
    
    return dataset

##
               
#==============================================================================

class MeshDataGenerator():
    
    def __init__(self,mesh,batch_size,sample_method,dataset_size):
        
        self.mesh = mesh
        
        self.batch_size = batch_size
        
        self.sample_method = sample_method
        
        self.dataset_size = dataset_size
        
        self.verts = mesh.vertices
        
        self.faces = mesh.faces
        
        self.stdev = 0.10
        
        self.ratio = 10.0
        
        self.scale = 60.0
        
        self.GetFaceAreaCDF()
        
    ##    
    
    #==============================================================================
    
    # Returns a cumulative distribution function of face areas (with min = 0, max = 1) in the same order as 'faces'
    
    def GetFaceAreaCDF(self):
                
        vector_ab = self.verts[self.faces[:,0],:] - self.verts[self.faces[:,1],:]
        
        vector_ac = self.verts[self.faces[:,0],:] - self.verts[self.faces[:,2],:]
        
        face_areas_abc = np.linalg.norm(np.cross(vector_ab,vector_ac),axis=1)
        
        face_areas_abc = face_areas_abc/ np.sum(face_areas_abc)
        
        self.face_areas_cdf = np.concatenate(([0],np.cumsum(face_areas_abc)))
    
    ##    

    #==============================================================================
    
    # Returns coordinates and signed distances of points randomly distributed within the bounding sphere of a mesh
    
    def UniformSampler(self,n_samples):
        
        bounding_sphere_radius = (1.0)*(np.ptp(self.mesh.bounding_sphere.bounds)/2.0)
        
        sample_points_3d = np.empty(shape=(0,3))
        
        remaining_points = n_samples
    
        while (remaining_points > 0):
            
            random_points = np.random.uniform(low=-bounding_sphere_radius,high=+bounding_sphere_radius,size=(remaining_points,3))
    
            random_points = random_points[np.linalg.norm(random_points,axis=1)<=bounding_sphere_radius]
            
            sample_points_3d = np.concatenate([sample_points_3d,random_points])
    
            remaining_points = n_samples - sample_points_3d.shape[0]
            
        ##
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
        
        return sample_points_3d, signed_distances
    
    ##    
    
    #==========================================================================
    
    # Returns coordinates and signed distances of points randomly sampled (with normally distributed offset) from the surface of a mesh
    # Barycentric coordinate system for sampling points from a triangle's vertices https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    
    def SurfaceSampler(self,n_samples):
    
        sample_face_indices = np.array(np.searchsorted(a=self.face_areas_cdf,v=np.random.rand(n_samples),side="right")-1)
    
        vertex_a = self.verts[self.faces[sample_face_indices,0],:]
        
        vertex_b = self.verts[self.faces[sample_face_indices,1],:]
        
        vertex_c = self.verts[self.faces[sample_face_indices,2],:]
        
        lambdas = np.random.rand(n_samples,3)
        
        lambdas = np.expand_dims((lambdas / np.expand_dims(np.sum(lambdas,axis=1),axis=-1)), axis=-1)
                
        sample_points_3d = (lambdas[:,0] * vertex_a) + (lambdas[:,1] * vertex_b) + (lambdas[:,2] * vertex_c)
        
        sample_points_3d = np.random.normal(loc=sample_points_3d,scale=self.stdev)
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
        
        return sample_points_3d, signed_distances
        
    ##
        
    #==========================================================================
    
    # Returns coordinates and signed distances of points randomly sampled (with normally distributed offset) from the nodes of a mesh
    
    def VerticeSampler(self,n_samples):
        
        sample_points_3d = self.verts[np.random.choice(self.verts.shape[0],n_samples),:]
    
        sample_points_3d = np.random.normal(loc=sample_points_3d,scale=self.stdev)
        
        signed_distances = np.expand_dims(a=trimesh.proximity.signed_distance(self.mesh,sample_points_3d),axis=-1)
    
        return sample_points_3d, signed_distances
    
    ##
    
    #==========================================================================
    
    # Returns coordinates and signed distances of points sampled using Davies et al.'s importance sampling strategy, for any given mesh
    # "On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes" -> https://arxiv.org/abs/2009.09808
    
    def ImportanceSampler(self,n_samples):
        
        uniform_points = self.UniformSampler(n_samples=(n_samples*self.ratio))
        
        signed_distances = trimesh.proximity.signed_distance(self.mesh,uniform_points)

        importance = np.exp(((-self.scale) * np.abs(signed_distances)))
                
        importance_cdf = np.concatenate(([0],np.cumsum(importance / np.sum(importance))))
        
        sample_point_indices = np.array(np.searchsorted(a=importance_cdf,v=np.random.rand(n_samples),side="right")-1)
        
        sample_points_3d = uniform_points[sample_point_indices,:]
        
        signed_distances = np.expand_dims(a=signed_distances[sample_point_indices,:],axis=-1)
        
        return sample_points_3d, signed_distances
    
    ##
    
    #==========================================================================
    
    # Could make this faster by moving the "if" statements to __init__ so it only does it once
    
    def GenerateBatch(self):
                            
        for batch in range(int(np.ceil(self.dataset_size/self.batch_size))):
        
            if self.sample_method == "uniform": 
                sample_points_3d,signed_distances = self.UniformSampler(self.batch_size)
            ##
            
            if self.sample_method == "surface": 
                sample_points_3d,signed_distances = self.SurfaceSampler(self.batch_size)
            ##
            
            if self.sample_method == "vertice": 
                sample_points_3d,signed_distances = self.VerticeSampler(self.batch_size)
            ##
            
            if self.sample_method == "importance": 
                sample_points_3d,signed_distances = self.ImportanceSampler(self.batch_size)
            ## 
                    
            yield tf.convert_to_tensor(sample_points_3d.astype("float32")),tf.convert_to_tensor(signed_distances.astype("float32"))
    
    ##
    
##
    
#==============================================================================

def MeshDatasetFromGenerator(mesh,batch_size,sample_method,dataset_size):
        
    data = MeshDataGenerator(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size)
    
    output_types = (tf.float32,tf.float32)
    
    output_shapes = (tf.TensorShape((batch_size,3)),tf.TensorShape((batch_size,1)))
    
    generator = lambda: data.GenerateBatch()

    dataset = tf.data.Dataset.from_generator(generator=generator,output_types=output_types,output_shapes=output_shapes)
            
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset.size = int(np.ceil(data.dataset_size/data.batch_size))
    dataset.batch_size = batch_size
    
    return dataset
    
    ##

#==============================================================================

class GridDataset():
    
    def __init__(self,mesh,batch_size,resolution,bbox_scale):
        
        self.mesh = mesh
        
        self.batch_size = batch_size
        
        self.resolution = resolution 
        
        self.bbox_scale = bbox_scale
        
        self.dataset_size = int(np.power(resolution,3))
        
    ##
    
    #==========================================================================
    
    def GenerateGrid(self):
        
        cx,cy,cz = self.mesh.bounding_box.centroid
    
        lx,ly,lz = self.mesh.bounding_box.extents
    
        xs = np.linspace((cx-(lx*self.bbox_scale/2)),(cx+(lx*self.bbox_scale/2)),self.resolution)
        
        ys = np.linspace((cy-(ly*self.bbox_scale/2)),(cy+(ly*self.bbox_scale/2)),self.resolution)
        
        zs = np.linspace((cz-(lz*self.bbox_scale/2)),(cz+(lz*self.bbox_scale/2)),self.resolution)
        
        grid = np.meshgrid(xs,ys,zs,indexing="ij")
    
        self.sample_points_3d = np.stack(grid,axis=-1).reshape((-1,3))
        
        return None
    
    ##
    
    #==========================================================================
    
    def GenerateData(self):
        
        self.signed_distances = np.empty(shape=(self.dataset_size,1))
        
        self.GenerateGrid()
            
        number_of_batches = int(np.ceil((self.dataset_size)/self.batch_size))
        
        for batch in range(number_of_batches):
            
            indices = [self.batch_size*batch, min(self.batch_size*(batch+1),self.dataset_size)]
                
            sample_points_3d_batch = self.sample_points_3d[slice(*indices),:]
            
            signed_distances_batch = trimesh.proximity.signed_distance(self.mesh,sample_points_3d_batch)
            
            self.signed_distances[slice(*indices),:] = np.expand_dims(a=signed_distances_batch,axis=-1)
            
            print(ProgressBar(current=(batch+1),end=number_of_batches),end="")
            
        ##
        
        return None
    
    ##        

#==============================================================================

mesh_filename = "/home/rms221/Documents/Compressive_Signed_Distance_Functions/ICML2021/neuralImplicitTools/data/bumpy-cube.obj"
mesh = trimesh.load(mesh_filename)
batch_size = 1024
resolution = 31
bbox_scale = 1.0

#==============================================================================

def MakeGridDataset(mesh,batch_size,resolution,bbox_scale,save_filepath,show=False): 
    
    grid_data = GridDataset(mesh=mesh,batch_size=batch_size,resolution=resolution,bbox_scale=bbox_scale)

    grid_data.GenerateData()
    
    np.save(file=save_filepath,arr=np.concatenate((grid_data.sample_points_3d,grid_data.signed_distances),axis=-1))
    
    sample_points_3d = tf.convert_to_tensor(grid_data.sample_points_3d.astype("float32"))
    
    signed_distances = tf.convert_to_tensor(grid_data.signed_distances.astype("float32"))
    
    dataset = tf.data.Dataset.from_tensor_slices((sample_points_3d,signed_distances))
    
    dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=grid_data.dataset_size,reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    dataset.dataset_size = grid_data.dataset_size
    dataset.size = len(dataset)
    dataset.batch_size = batch_size
    
    if show:
        grid_data.mesh.visual.face_colors = [255,0,0,128]
        points = trimesh.PointCloud(vertices=grid_data.sample_points_3d,colors=[0,0,255,128])
        trimesh.Scene([points,grid_data.mesh]).show(line_settings={'point_size':0.5,})
    ##
    
    return dataset

##

#==============================================================================

def LoadGridDataset(mesh,batch_size,resolution,bbox_scale,load_filepath,show=False):  
    
    grid_data = GridDataset(mesh=mesh,batch_size=batch_size,resolution=resolution)
    
    grid_data.sample_points_3d = np.load(load_filepath)[:,:-1]
    
    grid_data.signed_distances = np.load(load_filepath)[:,-1:]

    sample_points_3d = tf.convert_to_tensor(grid_data.sample_points_3d.astype("float32"))
    
    signed_distances = tf.convert_to_tensor(grid_data.signed_distances.astype("float32"))
    
    dataset = tf.data.Dataset.from_tensor_slices((sample_points_3d,signed_distances))
    
    dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=grid_data.dataset_size,reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size=batch_size,drop_remainder=False,num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    dataset.dataset_size = grid_data.dataset_size
    dataset.size = len(dataset)
    dataset.batch_size = batch_size
    
    if show:
        grid_data.mesh.visual.face_colors = [255,0,0,128]
        points = trimesh.PointCloud(vertices=grid_data.sample_points_3d,colors=[0,0,255,128])
        trimesh.Scene([points,grid_data.mesh]).show(line_settings={'point_size':0.5,})
    ##
    
    return dataset

##
    
#==============================================================================

def ProgressBar(current,end):

    if end < current: current = end

    progress = np.round((current/end)*25).astype(int)
    
    complete = (current/end)*100

    prog_bar = "[" + ("="*(progress)) + ">" + ("."*(25-progress)) + "]" + " {:5.1f}% Complete".format(complete)
    
    prog_bar = "\r{:30}{}".format("Progress:",prog_bar)
    
    return prog_bar

##

#==========================================================================
# Test 1

# dataset_size = 10000
# batch_size = 1024
# sample_method = "vertice"

# mesh_filename = "/home/rms221/Documents/Compressive_Signed_Distance_Functions/ICML2021/neuralImplicitTools/data/bumpy-cube.obj"
# mesh = trimesh.load(mesh_filename)
# tick = time.time()
# dataset = MakeDatasetFromGenerator(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size)    
# for batch in dataset: pass
# tock = time.time()
# print("Elapsed Time: {:.3}s".format(tock-tick))

# mesh_filename = "/home/rms221/Documents/Compressive_Signed_Distance_Functions/ICML2021/neuralImplicitTools/data/bumpy-cube.obj"
# mesh = trimesh.load(mesh_filename)
# tick = time.time()
# dataset = MakeDataset(mesh=mesh,batch_size=batch_size,sample_method=sample_method,dataset_size=dataset_size)
# for batch in dataset: pass
# tock = time.time()
# print("Elapsed Time: {:.3}s".format(tock-tick))

#==============================================================================
# Test 2

# mesh_filename = "/home/rms221/Documents/Compressive_Signed_Distance_Functions/ICML2021/neuralImplicitTools/data/bumpy-cube.obj"
# mesh = trimesh.load(mesh_filename)
# data = DataDataset(mesh=mesh,batch_size=10,sample_method="vertice",dataset_size=65)

# data.GenerateData()
# points = trimesh.PointCloud(vertices=data.sample_points_3d)
# trimesh.Scene([points]).show(flags={'wireframe':True,},line_settings={'line_width':1,'point_size':1})

#==============================================================================