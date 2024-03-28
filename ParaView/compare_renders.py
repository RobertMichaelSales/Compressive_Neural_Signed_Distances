""" Created: 20.02.2024  \\  Updated: 20.02.2024  \\   Author: Robert Sales """

#==============================================================================

import glob, os
from PIL import Image, ImageDraw, ImageFont

#==============================================================================

def CompareMatrix(experiments,independent_variable,savename=None):
                
    view_angles = ["orth","xp","yp","zp"]
    
    rows = len(view_angles)
    cols = len(experiments)
    
    metadata = {"target_compression_ratio" :("{:.2f}"   ,[]),
                "initial_learning_rate"    :("{:.2E}"   ,[]),
                "frequencies"              :("{:}"      ,[]),
                "hidden_layers"            :("{:}"      ,[]),
                "activation"               :("{:}"      ,[]),
                "bits_per_neuron"          :("{:}"      ,[]),
                "architecture"             :("{:}"      ,[]),
                "sample_method"            :("{:}"      ,[]),
                "dataset_size"             :("{:.0E}"   ,[]),
                "normalise"                :("{:}"      ,[]),}
    
    x_border_l = 300
    x_border_r = 0
    y_border_t = 100
    y_border_b = 0
       
    image_matrix = Image.new(mode='RGB',size=((2000*(cols+1)),(2000*rows)))  
    
    for col,experiment in enumerate(experiments):
        
        metadata["target_compression_ratio"][1].append(float(os.path.basename(experiment).split("_")[1][4:-1]))
        metadata["initial_learning_rate"][1].append(float(os.path.basename(experiment).split("_")[2][4:-1]))
        metadata["frequencies"][1].append(int(os.path.basename(experiment).split("_")[3][4:-1]))
        metadata["hidden_layers"][1].append(int(os.path.basename(experiment).split("_")[4][4:-1]))
        metadata["activation"][1].append(str(os.path.basename(experiment).split("_")[5][4:-1]).title())
        metadata["bits_per_neuron"][1].append(int(os.path.basename(experiment).split("_")[6][4:-1]))
        metadata["architecture"][1].append(str(os.path.basename(experiment).split("_")[7][4:-1]).title())
        metadata["sample_method"][1].append(str(os.path.basename(experiment).split("_")[8][4:-1]).title())
        metadata["dataset_size"][1].append(int(os.path.basename(experiment).split("_")[9][4:-1]))
        metadata["normalise"][1].append(str(True if str(os.path.basename(experiment).split("_")[8][1:-1]) == "NORM" else False).title())
        
        for row,view_angle in enumerate(view_angles):
            
            if col == 0:
                true_image_single = Image.open(os.path.join(experiment,"renders","true_sdf_{}.png".format(view_angle)))
                image_matrix.paste(im=true_image_single,box=((2000*(col)),(2000*row)))
            ##
            
            pred_image_single = Image.open(os.path.join(experiment,"renders","pred_sdf_{}.png".format(view_angle)))
            image_matrix.paste(im=pred_image_single,box=((2000*(col+1)),(2000*row)))
                        
        ##
    
    ##        
    
    image_border = Image.new(mode="RGB",size=(image_matrix.size[0]+(x_border_l+x_border_r),image_matrix.size[1]+(y_border_t+y_border_b)),color=(255,255,255))
    
    image_border.paste(im=image_matrix,box=(x_border_l,y_border_t))
    
    draw_text = ImageDraw.Draw(image_border)
    
    myFont = ImageFont.truetype('FreeMono.ttf',150)

    ##
    
    row_titles = view_angles
    col_titles = ["Reference"] + ["{}".format(metadata[independent_variable][0]).format(x) for x in metadata[independent_variable][1]]
      
    ##

    for col,col_title in enumerate(col_titles):
        
        draw_text.text(xy=((2000*col)+1000+x_border_l,65),text=col_title,font=myFont,fill=(0,0,0),anchor="mt",align="center",direction="ltr")
        
    ##
    
    for row,row_title in enumerate(row_titles):
        
        draw_text.text(xy=(65,(2000*row)+1000+y_border_t),text=row_title,font=myFont,fill=(0,0,0),anchor="lm",align="center",direction="ltr")
        
    ##

    if savename: 
        image_border.save(savename)
    else:
        image_border.show()
    ##
    
    return None

##

#==============================================================================

def OverlayImages():
    
    experiment = "/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(ELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"
    
    view_angle = "orth"
    
    for view_angle in ["orth","xp","yp","zp"]:
    
        true_image = Image.open(os.path.join(experiment,"renders","true_sdf_{}.png".format(view_angle)))
        true_image.putalpha(255)
        
        pred_image = Image.open(os.path.join(experiment,"renders","pred_sdf_{}.png".format(view_angle)))
        pred_image.putalpha(128)
        
        composite_image = Image.alpha_composite(true_image,pred_image)
        
        composite_image.show()
        
    ##
    
    return composite_image

##

#==============================================================================

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_armadillo_basic_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_armadillo_siren_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_domain0_basic_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_domain0_siren_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_domain1_basic_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(001)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(*)_(NORM)"))
# savename = "exp1_domain1_siren_dataset_size.png"
# independent_variable = "dataset_size"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

##

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_armadillo_basic_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_armadillo_siren_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_domain0_basic_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_domain0_siren_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_domain1_basic_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(002)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(*)_MDS(1000000)_(NORM)"))
# savename = "exp2_domain1_siren_sample_method.png"
# independent_variable = "sample_method"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

##

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(003)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(*)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# experiments = [x for x in experiments if not any(y.upper() in x for y in ["exponential","linear","selu","sigmoid"])]
# savename = "exp3_armadillo_basic_activations.png"
# independent_variable = "activation"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(003)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(*)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# experiments = [x for x in experiments if not any(y.upper() in x for y in ["exponential","linear","selu","sigmoid"])]
# savename = "exp3_domain0_basic_activations.png"
# independent_variable = "activation"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(003)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(*)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# experiments = [x for x in experiments if not any(y.upper() in x for y in ["exponential","linear","selu","sigmoid"])]
# savename = "exp3_domain1_basic_activations.png"
# independent_variable = "activation"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

##

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_armadillo_basic_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_armadillo_siren_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(GAUSS)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_armadillo_gauss_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain0_basic_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain0_siren_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(GAUSS)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain0_gauss_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain1_basic_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain1_siren_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(*)_NHL(008)_ACT(RELU)_BPN(032)_NNA(GAUSS)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain1_gauss_frequencies.png"
# independent_variable = "frequencies"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(RELU)_BPN(032)_NNA(*)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_armadillo_all_architectures.png"
# independent_variable = "architecture"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(RELU)_BPN(032)_NNA(*)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain0_all_architectures.png"
# independent_variable = "architecture"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(004)_TCR(0010.000000)_ILR(0.001000000)_PEF(016)_NHL(008)_ACT(RELU)_BPN(032)_NNA(*)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp4_domain1_all_architectures.png"
# independent_variable = "architecture"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

##

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_armadillo_basic_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_armadillo_siren_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_domain0_basic_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_domain0_siren_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_domain1_basic_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

# experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(005)_TCR(0010.000000)_ILR(0.001000000)_PEF(000)_NHL(*)_ACT(RELU)_BPN(032)_NNA(SIREN)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
# savename = "exp5_domain1_siren_hidden_layers.png"
# independent_variable = "hidden_layers"
# CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

##

experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/armadillo_mesh/EXP(006)_TCR(*)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
savename = "exp6_armadillo_basic_compression.png"
independent_variable = "target_compression_ratio"
CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_0_mesh/EXP(006)_TCR(*)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
savename = "exp6_domain0_basic_compression.png"
independent_variable = "target_compression_ratio"
CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

experiments = sorted(glob.glob("/Data/SDF_Compression_Experiments/domain_1_mesh/EXP(006)_TCR(*)_ILR(0.001000000)_PEF(000)_NHL(008)_ACT(RELU)_BPN(032)_NNA(BASIC)_DSM(VERTICE)_MDS(1000000)_(NORM)"))
savename = "exp6_domain1_basic_compression.png"
independent_variable = "target_compression_ratio"
CompareMatrix(experiments=experiments,independent_variable=independent_variable,savename=savename)

#==============================================================================