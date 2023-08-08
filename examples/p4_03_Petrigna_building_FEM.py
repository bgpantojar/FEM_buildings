import sys
sys.path.append("../src")
from FEM_generator import FEM_main

#p4_03 
data_folder = 'p4_03_Petrigna_building' 
im1 = ('CC0_3054','DJI_0965','DJI_0973','DJI_0985','DJI_0986','DJI_0998')
how2get_kp = 1 #0: detecting openings with CNN; 1:reading npy files with opening coordinates
path_builder_script = "../examples/p4_03_Petrigna_building_LOD3.py" #Path to file to call freecad builder
ctes=[.0, .2, .3, .3, .2] #ctes[0,1,2,3] -> opening regularization. ctes[4] = cte_fil4 -> filtering small detected openings
orientation_type = "pca"
approach = "C"
modal_analysis = True
display_EFM = True
FEM_main(data_folder, im1, path_builder_script, how2get_kp = how2get_kp, ctes = ctes, orientation_type = orientation_type, approach = approach, modal_analysis = modal_analysis, display_EFM = display_EFM)