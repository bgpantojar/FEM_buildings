import sys
sys.path.append("../src")
from FEM_generator import FEM_main

#p4_08 
data_folder = 'p4_08_Country_house_3' 
im1 = ('frame000450','frame002400','frame003330','frame004650')
how2get_kp = 1 #0: detecting openings with CNN; 1:reading npy files with opening coordinates
path_builder_script = "../examples/p4_08_Country_house_3_LOD3.py" #Path to file to call freecad builder
ctes=[.0, .3, .3, .3, .2] #ctes[0,1,2,3] -> opening regularization. ctes[4] = cte_fil4 -> filtering small detected openings
orientation_type = "main_normals"
approach = "C"
modal_analysis = True
display_EFM = True
FEM_main(data_folder, im1, path_builder_script, how2get_kp = how2get_kp, ctes = ctes, orientation_type = orientation_type, approach = approach, modal_analysis = modal_analysis, display_EFM = display_EFM)