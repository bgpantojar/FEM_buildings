import sys
sys.path.append("../src")
from FEM_generator import FEM_main

##p4_02 
data_folder = 'p4_02_Petrigna_school'  
im1 = ('DJI_0890','DJI_0892',"DJI_0898", 'DJI_0900','DJI_0909','DJI_0924','DJI_0928', 'DJI_0932', 'DJI_0956')
how2get_kp = 1 #0: detecting openings with CNN; 1:reading npy files with opening coordinates
path_builder_script = "../examples/p4_02_Petrigna_school_LOD3.py" #Path to file to call freecad builder
ctes=[.0, .3, .3, .3, 0.01]
orientation_type = "pca"
approach = "C"
modal_analysis = True
display_EFM = True
FEM_main(data_folder, im1, path_builder_script, how2get_kp = how2get_kp, ctes = ctes, orientation_type = orientation_type, approach = approach, modal_analysis = modal_analysis, display_EFM = display_EFM)