""" Created: 20.02.2024  \\  Updated: 20.02.2024  \\   Author: Robert Sales """

#==============================================================================

import glob, os
from PIL import Image, ImageDraw, ImageFont

#==============================================================================

def CompareMatrix():
    
    experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(elu)_BPN(032)_SIREN_*"))
    view_angles = ["orth","xp","yp","zp"]
    
    rows = len(view_angles)
    cols = len(experiments)
    
    row_titles = []
    col_titles = []
    
    x_border_l = 300
    x_border_r = 0
    y_border_t = 100
    y_border_b = 0
    
    image_matrix = Image.new(mode='RGB',size=((2000*cols),(2000*rows)))
    
    for col,experiment in enumerate(experiments):
        
        target_compression_ratio = float(os.path.basename(experiment).split("_")[1][4:-1])
        initial_learning_rate = float(os.path.basename(experiment).split("_")[2][4:-1])
        frequencies = int(os.path.basename(experiment).split("_")[3][4:-1])
        hidden_layers = int(os.path.basename(experiment).split("_")[4][4:-1])
        activation = str(os.path.basename(experiment).split("_")[5][4:-1])
        bits_per_neuron = int(os.path.basename(experiment).split("_")[6][4:-1])
        architecture = str(os.path.basename(experiment).split("_")[7][:])
        normalise = str(True if str(os.path.basename(experiment).split("_")[8][:]) == "NORM" else False)
        
        col_titles.append("Normalise: {}".format(normalise))
        
        for row,view_angle in enumerate(view_angles):
            
            image_single = Image.open(os.path.join(experiment,"renders","pred_sdf_{}.png".format(view_angle)))
            image_matrix.paste(im=image_single,box=((2000*col),(2000*row)))
            
            row_titles.append(view_angle)
            
        ##
    
    image_border = Image.new(mode="RGB",size=(image_matrix.size[0]+(x_border_l+x_border_r),image_matrix.size[1]+(y_border_t+y_border_b)),color=(255,255,255))
    
    image_border.paste(im=image_matrix,box=(x_border_l,y_border_t))
    
    draw_text = ImageDraw.Draw(image_border)
    
    myFont = ImageFont.truetype('FreeMono.ttf',150)
    
    for col,col_title in enumerate(col_titles):
        
        draw_text.text(xy=((2000*col)+1000+x_border_l,65),text=col_title,font=myFont,fill=(0,0,0),anchor="mt",align="center",direction="ltr")
        
    ##
    
    for row,row_title in enumerate(row_titles):
        
        draw_text.text(xy=(65,(2000*row)+1000+y_border_t),text=row_title,font=myFont,fill=(0,0,0),anchor="lm",align="center",direction="ltr")
        
    ##
    
    image_border.show()
    
    return None

##

#==============================================================================

def OverlayImages():
    
    experiment = "/Data/SDF_Compression_Experiments/armadillo/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(elu)_BPN(032)_SIREN_NORM"
    
    view_angle = "orth"
    
    for view_angle in ["orth","xp","yp","zp"]:
    
        true_image = Image.open(os.path.join(experiment,"renders","true_sdf_{}.png".format(view_angle)))
        true_image.putalpha(255)
        
        pred_image = Image.open(os.path.join(experiment,"renders","pred_sdf_{}.png".format(view_angle)))
        pred_image.putalpha(128)
        
        composite_image = Image.alpha_composite(true_image,pred_image)
        
        composite_image.show()
    
    return composite_image

##
