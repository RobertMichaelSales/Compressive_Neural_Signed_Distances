import numpy as np
import matplotlib.pyplot as plt
import trimesh

#==============================================================================

def Normalise(vector):
    
    vector = vector / np.linalg.norm(vector)
    
    return vector

##

#==============================================================================

class Viewer():
    
    def __init__(self,camera_origin,screen_normal,fov,resolution):
        
        self.camera_origin = camera_origin
        
        self.screen_normal = Normalise(screen_normal)
        
        self.fov = fov
        
        self.h_res = resolution[0]
        
        self.v_res = resolution[1]
        
    ##
        
    def GetPixelCentres(self):
    
        up_vector = np.array([0,1,0])
        
        lr_vector = ((Normalise(np.cross(self.screen_normal,up_vector)) * np.linalg.norm(self.screen_normal) * np.tan(self.fov/2) * 2.0) / self.v_res)

        tb_vector = ((Normalise(np.cross(self.screen_normal,lr_vector)) * np.linalg.norm(self.screen_normal) * np.tan(self.fov/2) * 2.0) / self.v_res)
        
        screen_points = np.stack([np.fromfunction(lambda i,j: self.camera_origin[k] + self.screen_normal[k] + ((i-(self.h_res/2))*lr_vector[k]) + ((j-(self.v_res/2))*tb_vector[k]), shape=(self.h_res+1,self.v_res+1)) for k in [0,1,2]],axis=-1)
                
        self.pixel_centres = np.multiply(0.5,(screen_points[:-1,:-1] + screen_points[ 1:, 1:]))
                
    ## 
    
    def Mesh(self):
        
        self.camera_mesh = trimesh.PointCloud(vertices=self.camera_origin.reshape((-1,3)))
        self.screen_mesh = trimesh.PointCloud(vertices=self.pixel_centres.reshape((-1,3)))
        
    ##
    
##

#==============================================================================

class Ray():
    
    def __init__(self,camera_origin,pixel_centre):
        
        self.camera_origin = camera_origin
        
        self.pixel_centre = pixel_centre
                
        self.ray_direction = Normalise(pixel_centre - camera_origin)

    ##
    
    def March(self,sdf):
        
        ray_in_bounds = True
        
        ray_is_active = True
        
        ray_parameter = 0.00
        
        sdf_tolerance = 0.01
        
        num_ray_steps = 0
        
        while(ray_in_bounds and ray_is_active and (num_ray_steps < 100)):
            
            ray_position = self.pixel_centre + (self.ray_direction * (1 + ray_parameter))
            
            value_at_ray = sdf(position=ray_position)
            
            if (np.abs(value_at_ray) <= sdf_tolerance):
                ray_is_active = False
                ray_hit = 1
                continue
            else:
                ray_is_active = True
                ray_hit = 0
            ##
            
            if np.linalg.norm(ray_position) >= 10:
                ray_in_bounds = False
                ray_hit = 0
                continue
            else:
                ray_in_bounds = True
                ray_hit = 0
            ##
            
            ray_parameter = ray_parameter + value_at_ray
            num_ray_steps = num_ray_steps + 1
            
        ##
        
        ray_depth = np.linalg.norm(ray_position - self.camera_origin)
                
        return ray_hit,ray_depth
        
    ##
    
##

#==============================================================================

class Sphere():
    
    def __init__(self,centre,radius):
        
        self.centre = centre
        
        self.radius = radius
        
    ##
    
    def GetSDF(self,position):
    
        value = np.linalg.norm(position - self.centre) - self.radius
        
        return value
    
    ##
    
    def Mesh(self):
        
        self.mesh = trimesh.primitives.Sphere(center=self.centre,radius=self.radius)
        
    ##
    
##

#==============================================================================

class Cuboid():
    
    def __init__(self,centre,corner):
        
        self.centre = centre
        
        self.corner = corner
        
    ##
    
    def GetSDF(self,position):
    
        value = np.linalg.norm(np.maximum((np.abs(position - self.centre) - self.corner),np.array([0.0,0.0,0.0])))
                
        return value
    
    ##
    
    def Mesh(self):
        
        self.mesh =  trimesh.primitives.Box(bounds=[self.centre - self.corner,self.centre + self.corner])
        
    ##
    
##

#==============================================================================

camera_origin = np.array([-3.0,-2.0,2.0])

# screen_normal = Normalise(np.array([+1.0,+1.0,-1.0]))

screen_normal = Normalise(-1 * camera_origin)

fov = np.pi/6

resolution = (120*2,100*2)

viewer = Viewer(camera_origin=camera_origin,screen_normal=screen_normal,fov=fov,resolution=resolution)

viewer.GetPixelCentres()

# sphere = Sphere(centre=np.array([0.0,0.0,0.0]),radius=1.0)

cuboid = Cuboid(centre=np.array([0.0,0.0,0.0]),corner=np.array([0.5,0.5,0.5]))

colour = np.zeros(shape=resolution)

for i in range(resolution[0]):
    
    for j in range(resolution[1]):
        
        ray = Ray(camera_origin=viewer.camera_origin,pixel_centre=viewer.pixel_centres[i,j])
        
        # ray_hit,ray_depth = ray.March(sdf=sphere.GetSDF)
        
        ray_hit,ray_depth = ray.March(sdf=cuboid.GetSDF)
        
        colour[i,j] = ray_hit * ray_depth
                
    ##
    
##

#==============================================================================

viewer.Mesh()
viewer.camera_mesh.visual.vertex_colors = [255,0,0,255]
viewer.screen_mesh.visual.vertex_colors = [0,255,0,255]

# sphere.Mesh()
# sphere.mesh.visual.face_colors = [0,0,255,128]
# scene = trimesh.Scene([sphere.mesh,viewer.camera_mesh,viewer.screen_mesh])

cuboid.Mesh()
cuboid.mesh.visual.face_colors = [255,0,0,128]
scene = trimesh.Scene([cuboid.mesh,viewer.camera_mesh,viewer.screen_mesh])

scene.show(flags={'wireframe':False},line_settings={'line_width':2.0,'point_size':10.0,})

#==============================================================================

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.imshow(np.flipud(colour.T),cmap="Greys",interpolation='nearest')
ax.axis("equal")
plt.show()

#==============================================================================

# screen_points = viewer.screen_points
# pixel_centres = viewer.pixel_centres

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(pixel_centres[:,:,0],pixel_centres[:,:,1],pixel_centres[:,:,2],s=2,marker='o',color='r',alpha=0.7)
# ax.scatter(camera_origin[0]    ,camera_origin[1]    ,camera_origin[2]    ,s=2,marker='s',color='g',alpha=0.7)

# ax.plot(xs=[camera_origin[0],screen_points[ 0, 0, 0]],ys=[camera_origin[1],screen_points[ 0, 0, 1]],zs=[camera_origin[2],screen_points[ 0, 0, 2]],color="green")
# ax.plot(xs=[camera_origin[0],screen_points[-1, 0, 0]],ys=[camera_origin[1],screen_points[-1, 0, 1]],zs=[camera_origin[2],screen_points[-1, 0, 2]],color="green")
# ax.plot(xs=[camera_origin[0],screen_points[ 0,-1, 0]],ys=[camera_origin[1],screen_points[ 0,-1, 1]],zs=[camera_origin[2],screen_points[ 0,-1, 2]],color="green")
# ax.plot(xs=[camera_origin[0],screen_points[-1,-1, 0]],ys=[camera_origin[1],screen_points[-1,-1, 1]],zs=[camera_origin[2],screen_points[-1,-1, 2]],color="green")

# ax.plot(xs=[screen_points[ 0, 0, 0],screen_points[-1, 0, 0]],ys=[screen_points[ 0, 0, 1],screen_points[-1, 0, 1]],zs=[screen_points[ 0, 0, 2],screen_points[-1, 0, 2]],color="green")
# ax.plot(xs=[screen_points[-1, 0, 0],screen_points[-1,-1, 0]],ys=[screen_points[-1, 0, 1],screen_points[-1,-1, 1]],zs=[screen_points[-1, 0, 2],screen_points[-1,-1, 2]],color="green")
# ax.plot(xs=[screen_points[-1,-1, 0],screen_points[ 0,-1, 0]],ys=[screen_points[-1,-1, 1],screen_points[ 0,-1, 1]],zs=[screen_points[-1,-1, 2],screen_points[ 0,-1, 2]],color="green")
# ax.plot(xs=[screen_points[ 0,-1, 0],screen_points[ 0, 0, 0]],ys=[screen_points[ 0,-1, 1],screen_points[ 0, 0, 1]],zs=[screen_points[ 0,-1, 2],screen_points[ 0, 0, 2]],color="green")

# # phi = np.linspace(0, np.pi, 32)
# # theta = np.linspace(0, 2*np.pi, 32)
# # phi, theta = np.meshgrid(phi, theta)
# # sphere.x = np.sin(phi) * np.cos(theta)
# # sphere.y = np.sin(phi) * np.sin(theta)
# # sphere.z = np.cos(phi)
# # ax.plot_surface(sphere.x,sphere.y,sphere.z,rstride=1,cstride=1,color='blue',alpha=0.7)

# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# ax.axis("equal")
# ax.view_init(90,270)
# plt.show()

#==============================================================================