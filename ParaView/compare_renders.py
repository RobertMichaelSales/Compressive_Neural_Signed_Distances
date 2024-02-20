""" Created: 20.02.2024  \\  Updated: 20.02.2024  \\   Author: Robert Sales """

import glob, os
from PIL import Image, ImageDraw, ImageFont

experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(elu)_BPN(032)_SIREN_*"))
view_angles = ["orth","xp","yp","zp"]

rows = len(view_angles)
cols = len(experiments)

row_titles = []
col_titles = []

comparison_image = Image.new(mode='RGB',size=((2000*cols),(2000*rows)))
draw_text = ImageDraw.Draw(comparison_image)
myFont = ImageFont.truetype('FreeMono.ttf',150)

for col,experiment in enumerate(experiments):
    
    target_compression_ratio = float(os.path.basename(experiment).split("_")[1][4:-1])
    initial_learning_rate = float(os.path.basename(experiment).split("_")[2][4:-1])
    frequencies = int(os.path.basename(experiment).split("_")[3][4:-1])
    hidden_layers = int(os.path.basename(experiment).split("_")[4][4:-1])
    activation = str(os.path.basename(experiment).split("_")[5][4:-1])
    bits_per_neuron = int(os.path.basename(experiment).split("_")[6][4:-1])
    architecture = str(os.path.basename(experiment).split("_")[7][:])
    normalise = str(True if str(os.path.basename(experiment).split("_")[8][:]) == "NORM" else False)
    
    col_titles.append(normalise)
    
    for row,view_angle in enumerate(view_angles):
        
        image = Image.open(os.path.join(experiment,"renders","pred_sdf_{}.png".format(view_angle)))
        comparison_image.paste(im=image,box=((2000*col),(2000*row)))
        
        row_titles.append(view_angle)
        
    ##
  
for col,col_title in enumerate(col_titles):
    
    draw_text.text(xy=((2000*col)+1000,65),text=col_title,font=myFont,fill=(255,0,0),anchor="mt",align="center",direction="ltr")
    
##

for row,row_title in enumerate(row_titles):
    
    draw_text.text(xy=(65,(2000*row)+1000),text=row_title,font=myFont,fill=(255,0,0),anchor="lm",align="center",direction="ttb")
      
comparison_image.show()