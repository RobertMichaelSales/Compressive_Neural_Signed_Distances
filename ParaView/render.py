""" Created: 13.02.2024  \\  Updated: 13.02.2024  \\   Author: Robert Sales """

# Note: Run this file from the command line using "pvpython filename.py"

#==============================================================================
# trace generated using paraview version 5.10.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#### reset paraview session
ResetSession()

#### import any other modules
import numpy as np
import os, json, glob
    
#==============================================================================
    
def Render(vts_filepath,img_savename,show_edges=False,render_zoom=1.0): 
    
    #==========================================================================
    # View & Camera
    
    # get active view
    render_view = GetActiveViewOrCreate('RenderView')
        
    #==========================================================================
    # Loading Data
    
    # create a new 'XML Structured Grid Reader'
    sdf_vts = XMLStructuredGridReader(registrationName='sdf.vts', FileName=[vts_filepath])
    
    # show data in view
    sdf_vtsDisplay = Show(sdf_vts, render_view, 'GeometryRepresentation')
    
    # extract the bounds of volume   
    bounds = sdf_vts.GetDataInformation().DataInformation.GetBounds()
    
    # determine bounding box centres
    centre_x = ((bounds[0]+bounds[1])/2)
    centre_y = ((bounds[2]+bounds[3])/2)
    centre_z = ((bounds[4]+bounds[5])/2)
    
    # determine bounding box lengths
    length_x = ((bounds[0]-bounds[1])/2)
    length_y = ((bounds[2]-bounds[3])/2)
    length_z = ((bounds[4]-bounds[5])/2)
    
    # determine bounding ball radius
    radius = np.linalg.norm([length_x,length_y,length_z])
    
    # hide data in view
    Hide(sdf_vts, render_view)
    
    #==========================================================================
    # Normalising
    
    # create a new 'Transform': translate
    sdf_vts = Transform(registrationName='Translate', Input=sdf_vts)
    sdf_vts.Transform = 'Transform'
    sdf_vts.Transform.Translate = [-centre_x,-centre_y,-centre_z]
    
    # create a new 'Transform': new_scale
    sdf_vts = Transform(registrationName='Scale', Input=sdf_vts)
    sdf_vts.Transform = 'Transform'
    sdf_vts.Transform.Scale = [1/radius,1/radius,1/radius]
    
    #==========================================================================
    # Make Contour
    
    # create a new 'Contour'
    contour = Contour(registrationName='contour', Input=sdf_vts)
    contour.ContourBy = ['POINTS', 'sdf']
    contour.Isosurfaces = [0]
    contour.PointMergeMethod = 'Uniform Binning'
    
    # show data in view
    contour_display = Show(contour, render_view, 'GeometryRepresentation')
    
    # show color bar/color legend
    contour_display.SetScalarBarVisibility(render_view, False)
    
    # turn off scalar coloring
    ColorBy(contour_display, None)
    
    if show_edges:
        
        # change representation type
        contour_display.SetRepresentationType('Surface With Edges')
        
        # properties modified on contour_display
        contour_display.RenderLinesAsTubes = 1
        
        # properties modified on contour_display
        contour_display.LineWidth = 0.25
        
    else:     
        
        # change representation type
        contour_display.SetRepresentationType('Surface')
        
    ##
    
    #==========================================================================
    # Environment
    
    # properties modified on render_view
    render_view.EnableRayTracing = 1
    
    # properties modified on render_view
    render_view.UseColorPaletteForBackground = 0
    
    # properties modified on render_view
    render_view.BackgroundColorMode = 'Skybox'
    
    # properties modified on render_view
    render_view.Background = [1.0, 1.0, 1.0]
    
    # properties modified on render_view
    render_view.OrientationAxesVisibility = 0
    
    # update the view to ensure updated data information
    render_view.Update()
    
    # reset view to fit data
    render_view.ResetCamera(True)
    
    #==========================================================================
    # Render 
    
    ## ISOMETRIC
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([2.1443472477763703, 1.4141082648137087, 2.1443472477763703])).tolist()
    render_view.CameraViewUp = [-0.29883623873011983, 0.906307787036650, -0.2988362387301198]
    render_view.CameraParallelScale = 0.6748609001214854
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_isom.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
    
    ## ORTHOGRAPIC
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([1.5721363955790437, 1.1444217043946554, 2.7230201135711063])).tolist()
    render_view.CameraViewUp = [-0.17101007166283452, 0.9396926207859084, -0.296198132726024]
    render_view.CameraParallelScale = 0.8660254037844386
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_orth.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
        
    
    ## X+ 
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([-4.50, 0.0, 0.0])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 1.0, 0.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_xp.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
        
    
    ## X-
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([+4.50, 0.0, 0.0])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 1.0, 0.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_xm.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
        
    ## Y+
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([0.0, -4.50, 0.0])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_yp.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
    
    ## Y-
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([0.0, 4.50, 0.0])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 0.0, 1.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_ym.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
        
    ## Z+ 
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([0.0, 0.0, -4.50])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 1.0, 0.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_zp.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
        
    ## Z-
    # set camera position
    render_view.CameraPosition = ((1/render_zoom)*np.array([0.0, 0.0, 4.50])).tolist()
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]
    render_view.CameraViewUp = [0.0, 1.0, 0.0]
    render_view.CameraViewAngle = 20.0
    render_view.CameraParallelScale = 0.9065242729925392
    
    # save as screenshot
    SaveScreenshot(img_savename.replace(".png","_zm.png"),render_view,ImageResolution=[2000,2000],TransparentBackground=False,CompressionLevel=0)
    
    # delete render view
    Delete(render_view)
    
##

#==============================================================================

if __name__=="__main__": 
    
    if (len(sys.argv) == 1):
        
        #======================================================================
        # This block will run in the event that this script is called in an IDE
        
        filepath    = "/home/rms221/Documents/Compressive_Neural_Signed_Distances/AuxFiles/outputs/armadillo/squashsdf"
        render_zoom = 1.0
          
    else: 

        #======================================================================
        # This block will run in the event that this script is run via terminal
        
        plotting_config = json.loads(sys.argv[1])
        
        filepath    = plotting_config["filepath"]
        render_zoom = plotting_config["render_zoom"]
        
    ##    
    
    #==========================================================================
        
    if not os.path.exists(os.path.join(filepath,"renders")): os.makedirs(os.path.join(filepath,"renders"))
    
    true_vts_filepath = os.path.join(filepath,"true_sdf.vts")
    true_img_savename = os.path.join(filepath,"renders","true_sdf.png")
    
    pred_vts_filepath = os.path.join(filepath,"pred_sdf.vts")
    pred_img_savename = os.path.join(filepath,"renders","pred_sdf.png")
    
    #==========================================================================

    # Check if the checkpoint file already exists
    if (len(glob.glob(true_img_savename.replace(".png","_*.png"))) != 8): 
    
        print("\nRendering '{}'".format(true_img_savename))
        Render(vts_filepath=true_vts_filepath,img_savename=true_img_savename,show_edges=False,render_zoom=1.0)
    
    else: 
        
        print("\nFiles '{}' already exist: skipping.".format(true_img_savename))
        
    ##
    
    # Check if the checkpoint file already exists
    if (len(glob.glob(pred_img_savename.replace(".png","_*.png"))) != 8): 
        
        print("\nRendering '{}'".format(pred_img_savename))
        Render(vts_filepath=pred_vts_filepath,img_savename=pred_img_savename,show_edges=False,render_zoom=1.0)
        
    else: 
        
        print("\nFiles '{}' already exist: skipping.".format(pred_img_savename))
        
    ##
    
else:pass
#==============================================================================
