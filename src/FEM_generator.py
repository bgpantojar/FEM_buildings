#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thr Sep 22 17:47:27 2022

This script uses camera data provided by meshroom
software located in the cameras.sfm file (StructureFromMotion folder)
See function added in CameraInit.py module (nodes) in meshroom.
It was implemented a script to save intrinsic and poses information
as dictionaries in json files.

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

@author: pantoja
"""
import numpy as np
import os
from projection_tools import *
from classes import *
from openings_projector import op_projector, load_sfm_json
import time
from scipy.spatial.distance import cdist
import shutil

import warnings
warnings.filterwarnings("ignore")

def fem_generator_solids(data_folder, dense, polyfit_path, sfm_json_path, images_path, bound_box, op_det_nn, keypoints_path, how2get_kp, im1, path_builder_script, ctes=[.05, .8, .1, .3], E=30e6, nu=0.2, rho=24.0, nmods=5, w_th=0.30, orientation_type="main_normals", modal_analysis=False):
    """

    """

    # Initial time
    t0 = time.time()

    # (I) Creating domain based on LOD2 - LOD2 regularization
    print("--- (I) Creating domain based on LOD2---")
    FEM = domain()
    #Load LOD2 obj file to domain - Assigning LOD2 to DADT domain
    FEM.load_LOD2(dense, polyfit_path)    
    #Initial cluster LOD2 triangle elements that are in the same plane and finde plane parameters
    FEM.LOD2.get_plane_clusters()
    #Decimate LOD2 model to reduce triangles number
    FEM.LOD2.decimate()
    #Update cluster LOD2 triangle elements that are in the same plane and find plane parameters
    FEM.LOD2.get_plane_clusters()    
    print("starting regulaization...")
    #Label planes into facades, floor, roof
    if orientation_type=="main_normals": #!
        FEM.LOD2.get_plane_labels(method='main_normals') 
    elif orientation_type=="pca":
        FEM.LOD2.get_plane_labels(method='pca') 
    #Prepare regularization
    FEM.LOD2.pre_regularize()
    #Regularized model
    FEM.LOD2.regularize_LOD(data_folder, type_transformation="facade_normals")
    #Update cluster LOD2 triangle elements that are in the same plane and find plane parameters
    FEM.LOD2.get_plane_clusters() 
    #Get plane labels once again as plane clusters change plane ids 
    if orientation_type=="main_normals":#!
        FEM.LOD2.get_plane_labels(method='main_normals') 
    elif orientation_type=="pca":
        FEM.LOD2.get_plane_labels(method='pca') 
    #prepare regularization once again as plane clusters change plane ids 
    FEM.LOD2.pre_regularize() 
    #Save LOD mesh
    FEM.LOD2.save_mesh(data_folder)
    #Save facade model
    FEM.LOD2.create_facade_model(data_folder)
    #Save roof model
    FEM.LOD2.create_roof_model(data_folder)
    #Save ground model
    FEM.LOD2.create_ground_model(data_folder)    
    #Get LOD2 boundary as lineset (to define facades, orientation, and simplify mesh?) 
    FEM.LOD2.get_boundary_line_set(plot=False)   
    #Get boundary for LOD2 planes (to help defining orientation maybe)
    FEM.LOD2.get_planes_contour()
    #Reading sfm information
    intrinsic, poses, structure = load_sfm_json(sfm_json_path)
    #Placing the LOD2 oriented and at the (0,0,0) initial coordinate and scaling it acording two points selected on one image
    FEM.LOD2.get_LOD_init(data_folder, im1, images_path, intrinsic, poses) 
    
    # Time checkpoint
    t1 = time.time()
    print(":::Finished part (I) in {}s:::".format(np.round(t1-t0,2)))
    
    print("--- (II) LOD3 generation based on image segmentation---")
    #Opening detector
    if how2get_kp == 0 or how2get_kp == 4:
        from opening_detector import main_op_detector
        opening_4points = main_op_detector(data_folder, images_path, bound_box, op_det_nn, cte_fil4 = ctes[4]) #!
    else:
        opening_4points = None    
    #Opening projector
    print("opening projection...")
    op_projector(FEM, data_folder, images_path, keypoints_path, intrinsic, poses, how2get_kp, im1, opening_4points, ctes=ctes)
    print("building LOD3...")  
    #Run the LOD3 builder inside FEM_generator
    cmmd = ("./LOD3_builder.sh {}".format(path_builder_script))
    os.system(cmmd)     
    
    # LOD3 processing
    print("LOD3 reading and regularization...")
    #Load LOD3 obj file to domain - Assigning LOD3 to FEM domain
    FEM.load_LOD3(data_folder)    
    #Initial cluster LOD3 triangle elements that are in the same plane and finde plane parameters
    FEM.LOD3.get_plane_clusters()
    #Decimate LOD3 model to reduce triangles number
    FEM.LOD3.decimate()
    #Update cluster LOD2 triangle elements that are in the same plane and find plane parameters
    FEM.LOD3.get_plane_clusters() 
    #Label planes into facades, floor, roof
    FEM.LOD3.get_plane_labels()
    #Save facade model
    FEM.LOD3.create_facade_model(data_folder)
    #Save roof model
    FEM.LOD3.create_roof_model(data_folder)
    #Save ground model
    FEM.LOD3.create_ground_model(data_folder)
    #Get LOD2 boundary as lineset (to define facades, orientation, and simplify mesh?) 
    FEM.LOD3.get_boundary_line_set(plot=False)
    #Get boundary for LOD3 planes (to help defining orientation maybe)
    FEM.LOD3.get_planes_contour()
    #Clone transformations from LOD2
    FEM.LOD3.clone_LOD2_T(FEM.LOD2)
    #Create LOD3 init
    FEM.LOD3.get_LOD_init(data_folder)
    #save mesh
    FEM.LOD3.save_mesh(data_folder)
    
    # Time checkpoint
    t2 = time.time()
    print(":::Finished part (II) in {}s:::".format(np.round(t2-t1,2)))

    print("---(III) Generation of Finite Element Models---")
    print("geting initial models according global coordinate system...")
    #Create init model. planes_init and openings init. Scaled and placed refered to (0,0,0)
    FEM.set_init_model()
    FEM.save_openings(data_folder, set_init=True, move_LOD_files=True)    
    print("Solid 3D FEM geometry...")
    #Generate FEM mesh using solid elements and gmsh
    FEM.generate_solid_FEM(data_folder, w_th = w_th)
    t3 = time.time()
    print(":::Finished part (III) in {}s:::".format(np.round(t3-t2,2)))
    
    ##Running modal analysis
    if modal_analysis:
        print("---(IV) Run modal analysis (MA) with Finite Element Model usin solid elements---")
        print("Running MA using solid elements in AMARU...")
        #Solids: using Amaru https://github.com/NumSoftware/Amaru.jl
        FEM.run_modal_analysis_solids(data_folder, E=E, nu=nu, rho=rho, nmods=nmods)
        t4 = time.time()
        print(":::Finished part (IV) in {}s:::".format(np.round(t4-t3,2)))

    return FEM


def FEM_generator_EFM(data_folder, approach, FEM, E, G, rho, fm, tau_fvm, fvlim, verif, driftT, driftPF, mu, Gc, beta_pier, beta_spandrel, w_th, nmods, modal_analysis=False, display_EFM = False):

    # Time checkpoint
    t4 = time.time()
    
    print("---(V) Generation of Finite Element Models - EFM ---")
    print("EFM discretization...")
    # Discretize LOD3 into piers,spandrels and nodes
    #Create facade lines
    FEM.create_LOD3_lines()
    #Create line intersections - FInd intersections of lines for each facade
    FEM.get_LOD3_line_intersections(plot=False)
    #Get the list of intersections valid for graph on each line. This to help create first line segments and later the graph
    FEM.get_LOD3_intersections_on_lines()
    #Create line segments and nodes(GRAph)
    FEM.get_LOD3_graph()
    #CREATE poligons with shortest graph path - Use the graph for first find the cycle basis of the graph and then the min cycle basis that are the poligons #!
    FEM.get_LOD3_poly_cells() 
    #Assign neighbours ids to the cells
    FEM.get_neighbour_cells_ids() 
    #Give the labels to poly cells. 
    FEM.get_cells_EFM_type(approach=approach)
    #save empty cells
    FEM.plot_cells_EFM(data_folder=data_folder, filled=False, save=True, name="empty_")
    print("end EFM discretization...")    
    t5 = time.time()
    print(":::Finished part (V) in {}s:::".format(np.round(t5-t4,2))) 


    print("---(VI) Generation of Finite Element Models - EFM ---")
    # Time for EFM generation
    t6 = time.time()

    #Plot cells and save them
    FEM.plot_elements_EFM(labels=True)
    plt.close()

    # Get macro elements cells. Create macroelements and nodes list objects
    
    #step0
    FEM.get_macro_elements_cells()
    #Save discretization obj files
    FEM.gen_obj_cells_EFM(data_folder)
    #Get the nodes positions to model in tremuri
    #label if node and element is contour or interior (contour up, down)
    for nodes_list_f in FEM.nodes_list:
        for n in nodes_list_f:        
            for pc in n.cells:
                if -1 in [pc.t, pc.b, pc.r, pc.l, pc.tl, pc.tr, pc.bl, pc.br]: #if any cell has no neighboor, is a contour
                    n.contour = True
    for elements_list_f in FEM.macro_elements_list:
        for me in elements_list_f:        
            for pc in me.cells:
                if -1 in [pc.t, pc.b, pc.r, pc.l, pc.tl, pc.tr, pc.bl, pc.br]: #if any cell has no neighboor, is a contour
                    me.contour = True
    
    #Finding t,b,l,r elements' neighboors
    #finding type of contour the node is
    #step1
    for i, nodes_list_f in enumerate(FEM.nodes_list):
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]
        #break
        for n in nodes_list_f:
            if n.contour:
                top_elements_id = [pc.t for pc in n.cells if fpc[pc.t].type!="n" and pc.t!=-1]
                bottom_elements_id = [pc.b for pc in n.cells if fpc[pc.b].type!="n" and pc.b!=-1]
                left_elements_id = [pc.l for pc in n.cells if fpc[pc.l].type!="n" and pc.l!=-1]
                right_elements_id = [pc.r for pc in n.cells if fpc[pc.r].type!="n" and pc.r!=-1]
                #finding type of contour the node is
                if len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)>0:
                    n.contour_type = "t"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)>0 and len(right_elements_id)>0:
                    n.contour_type = "b"
                elif len(top_elements_id)>0 and len(bottom_elements_id)>0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    n.contour_type = "l"
                elif len(top_elements_id)>0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    n.contour_type = "r"
                elif len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    n.contour_type = "tl"
                elif len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    n.contour_type = "tr"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    n.contour_type = "bl"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    n.contour_type = "br"

    #finding type of contour the element is
    for i, elements_list_f in enumerate(FEM.macro_elements_list):
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]
        #break
        for me in elements_list_f:
            if me.contour:
                top_elements_id = [pc.t for pc in me.cells if fpc[pc.t].type!=me.type and pc.t!=-1] #!
                bottom_elements_id = [pc.b for pc in me.cells if fpc[pc.b].type!=me.type and pc.b!=-1]
                left_elements_id = [pc.l for pc in me.cells if fpc[pc.l].type!=me.type and pc.l!=-1]
                right_elements_id = [pc.r for pc in me.cells if fpc[pc.r].type!=me.type and pc.r!=-1]
                #finding type of contour the node is
                if len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)>0:
                    me.contour_type = "t"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)>0 and len(right_elements_id)>0:
                    me.contour_type = "b"
                elif len(top_elements_id)>0 and len(bottom_elements_id)>0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    me.contour_type = "l"
                elif len(top_elements_id)>0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    me.contour_type = "r"
                elif len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    me.contour_type = "tl"
                elif len(top_elements_id)==0 and len(bottom_elements_id)>0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    me.contour_type = "tr"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)==0 and len(right_elements_id)>0:
                    me.contour_type = "bl"
                elif len(top_elements_id)>0 and len(bottom_elements_id)==0 and len(left_elements_id)>0 and len(right_elements_id)==0:
                    me.contour_type = "br"

    #Get the node coordinates for tremuri
    #facade planes
    #step2
    fps = [p for p in FEM.planes_init if p.type=='f']
    for i, nodes_list_f in enumerate(FEM.nodes_list):
        #facade poly cells
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]
        #Facade vertices
        facade_vertices = fps[i].plane_vertices_ordered
        min_x_facade = np.min(facade_vertices[:,0])
        max_x_facade = np.max(facade_vertices[:,0])
        for n in nodes_list_f:
            if not n.vanished: # to avoid the nodes that had cells whose type changed
                pc_coord = np.array([p.coord for pc in n.cells for p in pc.inter_list])
                n.c = np.array([np.mean(pc_coord[:,0]), np.mean(pc_coord[:,1])]) #If the node is regular rectangle. If it is polygon might need a change
                if n.contour:
                    if n.contour_type == "t":
                        n.coord = np.array([0.5*(np.min(pc_coord[:,0]+np.max(pc_coord[:,0]))), np.max(pc_coord[:,1])])
                    elif n.contour_type == "b":
                        n.coord = np.array([0.5*(np.min(pc_coord[:,0]+np.max(pc_coord[:,0]))), np.min(pc_coord[:,1])])
                    elif n.contour_type == "l":
                        n.coord = np.array([np.min(pc_coord[:,0]), 0.5*(np.min(pc_coord[:,1]+np.max(pc_coord[:,1])))])
                    elif n.contour_type == "r":
                        n.coord = np.array([np.max(pc_coord[:,0]), 0.5*(np.min(pc_coord[:,1]+np.max(pc_coord[:,1])))])
                    elif n.contour_type == "tl":
                        n.coord = np.array([np.min(pc_coord[:,0]), np.max(pc_coord[:,1])])
                    elif n.contour_type == "tr":
                        n.coord = np.array([np.max(pc_coord[:,0]), np.max(pc_coord[:,1])])
                    elif n.contour_type == "bl":
                        n.coord = np.array([np.min(pc_coord[:,0]), np.min(pc_coord[:,1])])
                    elif n.contour_type == "br":
                        n.coord = np.array([np.max(pc_coord[:,0]), np.min(pc_coord[:,1])])
                    elif n.contour_type == "":
                        n.coord = np.array([np.mean(pc_coord[:,0]), np.mean(pc_coord[:,1])])
                    #Assign node type. If they are contour and in the vertical walls of the facade extremes, then they are 3D. Otherwise 2D
                    if (np.abs(min_x_facade - n.coord[0]) < 5e-2) or (np.abs(max_x_facade - n.coord[0]) < 5e-2): #!
                        n.type = "3d"
                    else:
                        n.type = "2d"
                else:
                    n.coord = np.array([np.mean(pc_coord[:,0]), np.mean(pc_coord[:,1])])
                    n.type = "2d"

    #Get the center point of elements (c) for tremuri --> for those elements that will be split, this value will change
    #step3
    for i, macro_elements_list_f in enumerate(FEM.macro_elements_list):
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]
        for me in macro_elements_list_f:
            pc_coord = np.array([p.coord for pc in me.cells for p in pc.inter_list])
            me_max_x = np.max(pc_coord[:,0])
            me_min_x = np.min(pc_coord[:,0])
            me_max_y = np.max(pc_coord[:,1])
            me_min_y = np.min(pc_coord[:,1])
            me.c = np.array([0.5*(me_max_x + me_min_x), 0.5*(me_max_y + me_min_y)]) #If the element is regular rectangle. If it is polygon might need a change
            
    #Check if pier elements need to be split -- this in case there is a spandrel between nodes (or multiple nodes) below or upper the element
    #Select the conecting nodes of each element
    full_macro_elements_list = []
    for macro_elements_list_f in FEM.macro_elements_list: 
        full_macro_elements_list+=macro_elements_list_f
    #get nodes neighbours
    for i in range(len(FEM.macro_elements_list)):
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]
        for n in FEM.nodes_list[i]:
            n.t = np.unique(np.array([fpc[pc.t].macro_element_id for pc in n.cells if fpc[pc.t].type!="n"]))
            n.b = np.unique(np.array([fpc[pc.b].macro_element_id for pc in n.cells if fpc[pc.b].type!="n"]))
            n.l = np.unique(np.array([fpc[pc.l].macro_element_id for pc in n.cells if fpc[pc.l].type!="n"]))
            n.r = np.unique(np.array([fpc[pc.r].macro_element_id for pc in n.cells if fpc[pc.r].type!="n"]))
    full_node_elements_list = []
    for nodes_elements_list_f in FEM.nodes_list:
        full_node_elements_list +=nodes_elements_list_f

    #Elements and nodes ids for new ones
    new_macro_element_id = len(full_macro_elements_list)+1 #+1 as tremuri numbering starts at 1
    new_node_element_id = len(full_node_elements_list)+1 #+1 as tremuri numbering starts at 1

    #Create new nodes for piers are the bottom
    #step4
    for i, macro_elements_list_f in enumerate(FEM.macro_elements_list):
        for me in macro_elements_list_f:
            if me.type=="p" and me.contour:
                if me.contour_type in ["bl", "br", "b"]:
                    #Coord data
                    pc_coord = np.array([p.coord for pc in me.cells for p in pc.inter_list])
                    bl = np.array([np.min(pc_coord[:,0]), np.min(pc_coord[:,1])])
                    br = np.array([np.max(pc_coord[:,0]), np.min(pc_coord[:,1])])
                    #Create new node
                    new_node = node()
                    if me.contour_type=="b":
                        new_node.type = "2d"
                    else:
                        new_node.type = "3d" #as they will be surely conected to other facades
                    new_node.contour = True 
                    new_node.contour_type = me.contour_type
                    if me.contour_type=="bl":
                        new_node.coord = bl  #!
                    elif me.contour_type=="br":
                        new_node.coord = br #!
                    elif me.contour_type == "b":                
                        new_node.coord = 0.5*(bl+br)
                    new_node.c = new_node.coord
                    new_node.id = new_node_element_id
                    #Update node ids for new node
                    new_node_element_id+=1  
                    #Add new element to FEM.nodes_list
                    FEM.nodes_list[i]+=[new_node]
                    #Establish me neighboor to use it when creating elements 
                    me.b = new_node.id
    
    #update full nodes 
    full_node_elements_list = []
    for nodes_elements_list_f in FEM.nodes_list:
        full_node_elements_list +=nodes_elements_list_f
    #Elements and nodes ids for new ones
    new_node_element_id = len(full_node_elements_list)+1 #+1 as tremuri numbering starts at 1
    full_node_elements_ids = np.array([n.id for n in full_node_elements_list])

    #Go through macro-elements assigning b,h,c,[ni,nj] for later creation of tremuri geometry. If there are multiple nodes above and/or bottom of the pier
    #(left and/or right of spandrels), split the element as many times as necessary
    #step5
    for i, macro_elements_list_f in enumerate(FEM.macro_elements_list):
        fpc = [pc for pc in FEM.LOD3_poly_cells[i]]

        #if the number of macro-elements in the facade is 1, there is a single pier. Create 4 new nodes in the facade corners #!
        #and split the pier in two.
        if len(macro_elements_list_f)==1:
            #Create new nodes
            pc_coord = np.array([p.coord for pc in macro_elements_list_f[0].cells for p in pc.inter_list])
            pc_coord_z0 = pc_coord[np.where(pc_coord[:,1]==0)] 
            pier_cy = 0.5*(np.max(pc_coord[:,1]) + np.min(pc_coord[:,1]))
            initial_x = np.min(pc_coord_z0[:,0]) 
            pier_b = np.max(pc_coord_z0[:,0]) - np.min(pc_coord_z0[:,0]) 
            pier_h = np.max(pc_coord[:,1]) - np.min(pc_coord[:,1])
            bl = np.array([np.min(pc_coord_z0[:,0]), np.min(pc_coord[:,1])]) 
            tl = np.array([np.min(pc_coord_z0[:,0]), np.max(pc_coord[:,1])]) 
            br = np.array([np.max(pc_coord_z0[:,0]), np.min(pc_coord[:,1])]) 
            tr = np.array([np.max(pc_coord_z0[:,0]), np.max(pc_coord[:,1])]) 
            new_nodes_type = [["bl", bl],["tl", tl],["br", br],["tr", tr]]
            for j, nnt in enumerate(new_nodes_type):
                new_node = node()
                new_node.type = "3d" #as they will be surely conected to other facades
                new_node.contour = True 
                new_node.contour_type = nnt[0]
                new_node.coord = nnt[1]
                new_node.c = nnt[1]
                new_node.id = new_node_element_id
                #Update node ids for new node
                new_node_element_id+=1  
                #Add new element to FEM.nodes_list
                FEM.nodes_list[i]+=[new_node]
            #Create new elements
            for j in range(2):
                new_me = macro_element()
                new_me.type = "p"
                new_me.id = new_macro_element_id                    
                new_me.b = pier_b/2
                new_me.h = pier_h
                new_me.c = [initial_x + pier_b/4 + j*pier_b/2, pier_cy]
                n_i = FEM.nodes_list[i][2*j].id
                n_j = FEM.nodes_list[i][2*j+1].id
                new_me.ij_nodes = [n_i, n_j]
                
                #Add new element to the FEM.macro_elements_list[i]
                FEM.macro_elements_list[i] += [new_me]

                #update me ids for next new_me
                new_macro_element_id+=1
            #Activate split flag to initial element to avoid its creation in geo file
            FEM.macro_elements_list[i][0].split = True
        else:
            #for me in copy.deepcopy(macro_elements_list_f):
            for k in range(len(macro_elements_list_f)):
                #me poly cells vertices coordinates --> to define, c,h,b
                pc_coord = np.array([p.coord for pc in macro_elements_list_f[k].cells for p in pc.inter_list])
                #Take all the facade nodes. In case there is an element that doesnt have node to connect using node cells, conect to the closest node
                f_nodes = [n for n in FEM.nodes_list[i]]
                
                #for piers
                if macro_elements_list_f[k].type == 'p':
                    #Getting macroelement  top and bottom neighbours (they will be ordered accordingly previous lines)
                    top_nodes_id = np.unique(np.array([fpc[pc.t].macro_element_id for pc in macro_elements_list_f[k].cells if fpc[pc.t].type=="n" and pc.t!=-1])) #to be different than -1 the id
                    bottom_nodes_id = np.unique(np.array([fpc[pc.b].macro_element_id for pc in macro_elements_list_f[k].cells if fpc[pc.b].type=="n" and pc.b!=-1]))
                    if macro_elements_list_f[k].contour_type in ["bl", "br", "b"]:
                        #Add to the bottom nodes those created bcs the pier is at the bottom contour
                        bottom_nodes_id = np.concatenate((bottom_nodes_id, np.array([macro_elements_list_f[k].b]))).astype('int')
                    #if there are more than 1 nodes above or bottom, split.. otherwise just adde me features ij, b and c
                    if len(top_nodes_id)>1 or len(bottom_nodes_id)>1:
                        top_nodes_coordx = np.array([full_node_elements_list[np.where(full_node_elements_ids==j)[0][0]].coord[0] for j in top_nodes_id])#-1 as ids and python indx do not match
                        bottom_nodes_coordx = np.array([full_node_elements_list[np.where(full_node_elements_ids==j)[0][0]].coord[0] for j in bottom_nodes_id])
                        #Organize the nodes from left to right (this to connect piers among closest nodes of oposite sides)
                        order_id_top = np.argsort(top_nodes_coordx)
                        order_id_bottom = np.argsort(bottom_nodes_coordx)
                        top_nodes_id = top_nodes_id[order_id_top]
                        bottom_nodes_id = bottom_nodes_id[order_id_bottom]
                        top_nodes_coordx = top_nodes_coordx[order_id_top]
                        bottom_nodes_coordx = bottom_nodes_coordx[order_id_bottom]
                        #x coordinate where piers will split
                        split_pier_x = []
                        if len(top_nodes_id)>=len(bottom_nodes_id):
                            for j in range(len(top_nodes_id)-1):
                                n = full_node_elements_list[np.where(full_node_elements_ids==top_nodes_id[j])[0][0]]
                                s = full_macro_elements_list[n.r[0]-1]                            
                                split_pier_x.append(s.c[0])
                        else:
                            for j in range(len(bottom_nodes_id)-1):
                                n = full_node_elements_list[np.where(full_node_elements_ids==bottom_nodes_id[j])[0][0]]
                                s = full_macro_elements_list[n.r[0]-1]                            
                                split_pier_x.append(s.c[0])
                        #Creating extra me and assigning their features
                        pier_h = np.max(pc_coord[:,1]) - np.min(pc_coord[:,1]) #pier height
                        pier_cy = 0.5*(np.max(pc_coord[:,1]) + np.min(pc_coord[:,1])) #y coord of the pier center
                        initial_x = np.min(pc_coord[:,0])
                        last_x = np.max(pc_coord[:,0]) #to create last element
                        for jj, new_me_x in enumerate(split_pier_x+[last_x]):
                            new_me = macro_element()
                            new_me.type = macro_elements_list_f[k].type
                            new_me.id = new_macro_element_id                    
                            new_me.b = new_me_x - initial_x
                            new_me.h = pier_h
                            new_me.c = np.array([.5*(new_me_x + initial_x), pier_cy])
                            dist2top_nodes = np.abs(top_nodes_coordx-new_me.c[0])
                            dist2bottom_nodes = np.abs(bottom_nodes_coordx-new_me.c[0])
                            #Select the closest node in the direction of fewer nodes. Select the ordered node in the direction of more nodes.
                            if len(top_nodes_id)<len(bottom_nodes_id): 
                                n_i = bottom_nodes_id[np.argsort(bottom_nodes_coordx)[jj]]
                                n_j = top_nodes_id[np.argmin(dist2top_nodes)]
                            elif len(top_nodes_id)>len(bottom_nodes_id):
                                n_i = bottom_nodes_id[np.argmin(dist2bottom_nodes)]
                                n_j = top_nodes_id[np.argsort(top_nodes_coordx)[jj]]
                            else:
                                n_i = bottom_nodes_id[np.argsort(bottom_nodes_coordx)[jj]]
                                n_j = top_nodes_id[np.argsort(top_nodes_coordx)[jj]]
                            new_me.ij_nodes = [n_i, n_j]
                            #Add new element to the FEM.macro_elements_list[i]
                            FEM.macro_elements_list[i] += [new_me]
                            #update me ids and initialx for next new_me
                            new_macro_element_id+=1
                            initial_x = new_me_x
                        #Deactivate me as it was split. This to avoid generating it during geo file creation
                        macro_elements_list_f[k].split = True

                    else:
                        macro_elements_list_f[k].b = np.max(pc_coord[:,0]) - np.min(pc_coord[:,0])
                        macro_elements_list_f[k].h = np.max(pc_coord[:,1]) - np.min(pc_coord[:,1])
                        if len(bottom_nodes_id)>0: #If there is not node, connect to the closest at the bottom
                            n_i = bottom_nodes_id[0]
                        else:
                            f_nodes_bottom = [n for n in f_nodes if not n.vanished] #avoid changed nodes
                            f_nodes_bottom = [n for n in f_nodes_bottom if (n.coord[1]-macro_elements_list_f[k].c[1])<0]
                            f_nodes_bottom_coord = np.array([n.coord for n in f_nodes_bottom])
                            dist2nodes_bottom = cdist(macro_elements_list_f[k].c.reshape((1,2)), f_nodes_bottom_coord)
                            n_i = f_nodes_bottom[np.argmin(dist2nodes_bottom)].id
                        if len(top_nodes_id)>0:
                            n_j = top_nodes_id[0]
                        else:
                            f_nodes_top = [n for n in f_nodes if not n.vanished] #len(n.cells)>0 avoid changed cells
                            f_nodes_top = [n for n in f_nodes_top if ((n.coord[1]-macro_elements_list_f[k].c[1])>0)]
                            f_nodes_top_coord = np.array([n.coord for n in f_nodes_top])
                            dist2nodes_top = cdist(macro_elements_list_f[k].c.reshape((1,2)), f_nodes_top_coord)
                            n_j = f_nodes_top[np.argmin(dist2nodes_top)].id
                        macro_elements_list_f[k].ij_nodes = [n_i, n_j]
                
                #for spandrels
                elif macro_elements_list_f[k].type == 's':
                    #Getting macroelement  top and bottom neighbours (they will be ordered accordingly previous lines)
                    left_nodes_id = np.unique(np.array([fpc[pc.l].macro_element_id for pc in macro_elements_list_f[k].cells if fpc[pc.l].type=="n" and pc.l!=-1]))
                    right_nodes_id = np.unique(np.array([fpc[pc.r].macro_element_id for pc in macro_elements_list_f[k].cells if fpc[pc.r].type=="n" and pc.r!=-1]))
                    #if there are more than 1 nodes above or bottom, split.. otherwise just adde me features ij, b and c
                    if len(left_nodes_id)>1 or len(right_nodes_id)>1:
                        left_nodes_coordy = np.array([full_node_elements_list[np.where(full_node_elements_ids==j)[0][0]].coord[1] for j in left_nodes_id])#-1 as ids and python indx do not match
                        right_nodes_coordy = np.array([full_node_elements_list[np.where(full_node_elements_ids==j)[0][0]].coord[1] for j in right_nodes_id])
                        #Organize the nodes from left to right (this to connect piers among closest nodes of oposite sides)
                        order_id_left = np.argsort(left_nodes_coordy)
                        order_id_right = np.argsort(right_nodes_coordy)
                        left_nodes_id = left_nodes_id[order_id_left]
                        right_nodes_id = right_nodes_id[order_id_right]
                        left_nodes_coordy = left_nodes_coordy[order_id_left]
                        right_nodes_coordy = right_nodes_coordy[order_id_right]
                        #y coordinate where spandrels will split
                        split_spandrel_y = []
                        if len(left_nodes_id)>=len(right_nodes_id):
                            for j in range(len(left_nodes_id)-1):
                                n = full_node_elements_list[np.where(full_node_elements_ids==left_nodes_id[j])[0][0]]
                                p = full_macro_elements_list[n.t[0]-1]
                                split_spandrel_y.append(p.c[1])
                        else:
                            for j in range(len(right_nodes_id)-1):
                                n = full_node_elements_list[np.where(full_node_elements_ids==right_nodes_id[j])[0][0]]
                                p = full_macro_elements_list[n.t[0]-1]
                                split_spandrel_y.append(p.c[1])
                        #Creating extra me and assigning their features
                        spandrel_h = np.max(pc_coord[:,0]) - np.min(pc_coord[:,0]) #spandrel height -> as it change the horientation, height is horizontal
                        spandrel_cx = 0.5*(np.max(pc_coord[:,0]) + np.min(pc_coord[:,0])) #x coord of the spandrel center
                        initial_y = np.min(pc_coord[:,1])
                        last_y = np.max(pc_coord[:,1]) #to create last element
                        for jj, new_me_y in enumerate(split_spandrel_y+[last_y]):
                            new_me = macro_element()
                            new_me.type = macro_elements_list_f[k].type
                            new_me.id = new_macro_element_id                    
                            new_me.b = new_me_y - initial_y #note that b in spandrel is vertical as horientation change in relation to piers
                            new_me.h = spandrel_h
                            new_me.c = np.array([spandrel_cx, .5*(new_me_y + initial_y)])
                            dist2left_nodes = np.abs(left_nodes_coordy-new_me.c[1])
                            dist2right_nodes = np.abs(right_nodes_coordy-new_me.c[1])
                            #Select the closest node in the direction of fewer nodes. Select the ordered node in the direction of more nodes.
                            if len(left_nodes_id)<len(right_nodes_id): 
                                n_i = left_nodes_id[np.argmin(dist2left_nodes)]
                                n_j = right_nodes_id[np.argsort(right_nodes_coordy)[jj]]
                            elif len(left_nodes_id)>len(right_nodes_id):
                                n_i = left_nodes_id[np.argsort(left_nodes_coordy)[jj]]
                                n_j = right_nodes_id[np.argmin(dist2right_nodes)]
                            else:
                                n_i = left_nodes_id[np.argsort(left_nodes_coordy)[jj]]
                                n_j = right_nodes_id[np.argsort(right_nodes_coordy)[jj]]
                            new_me.ij_nodes = [n_i, n_j]
                            #Add new element to the FEM.macro_elements_list[i]
                            FEM.macro_elements_list[i] += [new_me]
                            #update me ids and initialx for next new_me
                            new_macro_element_id+=1
                            initial_y = new_me_y
                        #Deactivate me as it was split. This to avoid generating it during geo file creation
                        macro_elements_list_f[k].split = True
                    else:
                        macro_elements_list_f[k].b = np.max(pc_coord[:,1]) - np.min(pc_coord[:,1])
                        macro_elements_list_f[k].h = np.max(pc_coord[:,0]) - np.min(pc_coord[:,0])
                        if len(left_nodes_id)>0:
                            n_i = left_nodes_id[0]
                        else:
                            f_nodes_left = [n for n in f_nodes if not n.vanished ] #avoid nodes changed
                            f_nodes_left = [n for n in f_nodes_left if (n.coord[0]-macro_elements_list_f[k].c[0])<0]
                            f_nodes_left_coord = np.array([n.coord for n in f_nodes_left])
                            dist2nodes_left = cdist(macro_elements_list_f[k].c.reshape((1,2)), f_nodes_left_coord)
                            n_i = f_nodes_left[np.argmin(dist2nodes_left)].id
                        if len(right_nodes_id)>0:
                            n_j = right_nodes_id[0]
                        else:
                            f_nodes_right = [n for n in f_nodes if not n.vanished] #avoid nodes changed
                            f_nodes_right = [n for n in f_nodes_right if (n.coord[0]-macro_elements_list_f[k].c[0])>0]
                            f_nodes_right_coord = np.array([n.coord for n in f_nodes_right])
                            dist2nodes_right = cdist(macro_elements_list_f[k].c.reshape((1,2)), f_nodes_right_coord)
                            n_j = f_nodes_right[np.argmin(dist2nodes_right)].id

                        macro_elements_list_f[k].ij_nodes = [n_i, n_j]


    #Creating wall elements
    FEM.walls_list = []
    fps = [p for p in FEM.planes_init if p.type=='f']
    fps_ids = np.array([p.id for p in fps])
    #Ground vertices from facade planes
    fp_ground_pts = []
    for p in fps:
        p_ground = (p.plane_vertices_ordered[np.where(p.plane_vertices_ordered[:,1]==0)[0]][:,0]).reshape((-1,1))
        if p.parallel_to==1:
            fp_ground_pts+=[np.concatenate((p_ground, np.ones((2,1))*p.third_coord), axis=1)]
        elif p.parallel_to==0:
            fp_ground_pts+=[np.concatenate((np.ones((2,1))*p.third_coord, p_ground), axis=1)]
    fp_ground_pts = np.array(fp_ground_pts)
    #Ground vertices from ground plane. This defines the order of the walls
    gp = [p for p in FEM.planes_init if p.type=='g'][0] #!
    walls_vertices = np.concatenate((gp.plane_vertices_ordered, gp.plane_vertices_ordered[0].reshape((-1,2))), axis=0)
    for i in range(len(walls_vertices)-1):
        new_wall = wall()
        new_wall.id = i+1 
        w_vertices = walls_vertices[i:i+2]
        new_wall.init_coord = w_vertices[0]
        new_wall.end_coord = w_vertices[1]
        #Check the ground vertices given by the facade planes and ground plane to assing the wall.plane_id
        for j, pts in enumerate(fp_ground_pts):
            if ((new_wall.init_coord==pts[0]).all() and (new_wall.end_coord==pts[1]).all()) or ((new_wall.init_coord==pts[1]).all() and (new_wall.end_coord==pts[0]).all()):
                new_wall.plane_id = fps[j].id
                new_wall.parallel_to = fps[j].parallel_to
                new_wall.third_coord = fps[j].third_coord
                new_wall.nodes = FEM.nodes_list[np.where(fps_ids==new_wall.plane_id)[0][0]]
                new_wall.elements = FEM.macro_elements_list[np.where(fps_ids==new_wall.plane_id)[0][0]]
                new_wall.openings = FEM.openings_list[np.where(fps_ids==new_wall.plane_id)[0][0]]
                break
        
        #Select the wall.coord as the origin of local coord
        if new_wall.parallel_to==0:
            new_wall.coord = w_vertices[np.argmin(w_vertices[:,1])]
            new_wall.angle = 90 
        elif new_wall.parallel_to==1:
            new_wall.coord = w_vertices[np.argmin(w_vertices[:,0])]
            new_wall.angle = 0 

        FEM.walls_list.append(new_wall)

    #Getting redundant nodes. 3d nodes are connected to another 3d node. One of them is redundant keeping just the another to create the geometry file
    #The walls are ordered according connection then for n walls the conection is 1-2-3-4-5-...-n-1
    #Select the conecting nodes of each element. Here if there are new nodes created, it is necessary to create new piers in case of a wall without openings
    #or rigid nodes in case both walls have openings.
    #new ids for elements and nodes to create new ones when required
    #step6
    full_macro_elements_list = []
    for macro_elements_list_f in FEM.macro_elements_list: 
        full_macro_elements_list+=macro_elements_list_f
    full_node_elements_list = []
    for nodes_elements_list_f in FEM.nodes_list:
        full_node_elements_list +=nodes_elements_list_f
    new_macro_element_id = len(full_macro_elements_list)+1 #+1 as tremuri numbering starts at 1
    new_node_element_id = len(full_node_elements_list)+1 #+1 as tremuri numbering starts at 1
    for i, w in enumerate(FEM.walls_list):
        #break
        if i<len(FEM.walls_list)-1:
            w2 = FEM.walls_list[i+1]
        else:
            w2 = FEM.walls_list[0]
        
        #Getting 3d coordinates of 3d nodes to get correspondences
        #w
        nodes_3d_w = np.array([n for n in w.nodes if n.type=='3d'])
        nodes_3d_w_coord = np.array([n.coord for n in w.nodes if n.type=='3d'])
        dir_w_id = np.array(range(3))
        dir_w_id = np.delete(dir_w_id,w.parallel_to)
        nodes_3d_w_coord3d = np.zeros((len(nodes_3d_w_coord), 3))
        nodes_3d_w_coord3d[:,dir_w_id] = nodes_3d_w_coord
        nodes_3d_w_coord3d[:,w.parallel_to] = w.third_coord
        min_x_w = np.min(nodes_3d_w_coord[:,0])

        #w2
        nodes_3d_w2 = np.array([n for n in w2.nodes if n.type=='3d'])
        nodes_3d_w2_coord = np.array([n.coord for n in w2.nodes if n.type=='3d'])
        dir_w2_id = np.array(range(3))
        dir_w2_id = np.delete(dir_w2_id,w2.parallel_to)
        nodes_3d_w2_coord3d = np.zeros((len(nodes_3d_w2_coord), 3))
        nodes_3d_w2_coord3d[:,dir_w2_id] = nodes_3d_w2_coord
        nodes_3d_w2_coord3d[:,w2.parallel_to] = w2.third_coord
        min_x_w2 = np.min(nodes_3d_w2_coord[:,0])
        
        #The cdist for the [x,y] coordinates should be nule for those points that are over same wall edge
        dist_plan_w_w2_nodes = cdist(nodes_3d_w_coord3d[:,:2], nodes_3d_w2_coord3d[:,:2])
        #w
        nodes_corresp_w = nodes_3d_w[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[0])] #!
        nodes_corresp_w_coord3d = nodes_3d_w_coord3d[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[0])]
        nodes_corresp_w_coord = nodes_3d_w_coord[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[0])]
        #w2
        nodes_corresp_w2 = nodes_3d_w2[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[1])] #!
        nodes_corresp_w2_coord3d = nodes_3d_w2_coord3d[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[1])]
        nodes_corresp_w2_coord = nodes_3d_w2_coord[np.unique(np.where(dist_plan_w_w2_nodes<5e-2)[1])]
        #The correspondences are those with closest distances. If the number of nodes does not match, it needs to create a new (or new nodes) in the wall with less nodes
        #Initial correspondences are make using the min posible e.g., w=2nodes, w2=3nodes -> 2 correspondences made
        if len(nodes_corresp_w)>len(nodes_corresp_w2):
            dist_plan_w2_w_nodes_corresp = cdist(nodes_corresp_w2_coord3d, nodes_corresp_w_coord3d) 
            node_corresp_index_sort = np.argsort(dist_plan_w2_w_nodes_corresp, axis=1)
            node_corresp_index = node_corresp_index_sort[:,0]
            values, counts = np.unique(node_corresp_index_sort[:,0], return_counts=True)
            repeated_redundant = values[np.where(counts>1)]
            for rr in repeated_redundant:
                ind_rep  = np.where(node_corresp_index==rr)[0]
                if dist_plan_w2_w_nodes_corresp[ind_rep[0], rr]<dist_plan_w2_w_nodes_corresp[ind_rep[1], rr]:
                    node_corresp_index[ind_rep[1]] = node_corresp_index_sort[ind_rep[1]][1]
                else:
                    node_corresp_index[ind_rep[0]] = node_corresp_index_sort[ind_rep[0]][1]
            for j, n in enumerate(nodes_corresp_w2):
                n.redundant = True
                n.redundant_id = nodes_corresp_w[node_corresp_index[j]].id
                nodes_corresp_w[node_corresp_index[j]].redundant_id = n.id
        else:
            dist_plan_w_w2_nodes_corresp = cdist(nodes_corresp_w_coord3d, nodes_corresp_w2_coord3d) 
            node_corresp_index_sort = np.argsort(dist_plan_w_w2_nodes_corresp, axis=1)
            node_corresp_index = node_corresp_index_sort[:,0]
            values, counts = np.unique(node_corresp_index_sort[:,0], return_counts=True)
            repeated_redundant = values[np.where(counts>1)]
            for rr in repeated_redundant:
                ind_rep  = np.where(node_corresp_index==rr)[0]
                if dist_plan_w_w2_nodes_corresp[ind_rep[0], rr]<dist_plan_w_w2_nodes_corresp[ind_rep[1], rr]:
                    node_corresp_index[ind_rep[1]] = node_corresp_index_sort[ind_rep[1]][1]
                else:
                    node_corresp_index[ind_rep[0]] = node_corresp_index_sort[ind_rep[0]][1]
            for j, n in enumerate(nodes_corresp_w):
                n.redundant_id = nodes_corresp_w2[node_corresp_index[j]].id
                nodes_corresp_w2[node_corresp_index[j]].redundant = True
                nodes_corresp_w2[node_corresp_index[j]].redundant_id = n.id


        #If number of nodes correspondences for w and w2 are different, then needs to create nodes and split piers
        if len(nodes_corresp_w)!=len(nodes_corresp_w2):
            #Create new node(s) in the wall that has less nodes 
            new_nodes_wi = []
            if len(nodes_corresp_w)>len(nodes_corresp_w2):
                for n, n_coord in zip(nodes_corresp_w, nodes_corresp_w_coord3d):
                    if n.redundant_id==-1:
                        new_node = node()
                        new_node.type = "3d" #as they will be surely conected to other facades
                        new_node.contour = True 
                        new_node.coord = n_coord[dir_w2_id]
                        new_node.c = n_coord[dir_w2_id]
                        if new_node.c[0]==min_x_w2:
                            new_node.contour_type = "l"
                        else:
                            new_node.contour_type = "r"
                        new_node.id = new_node_element_id
                        #Update node ids for new node
                        new_node_element_id+=1  
                        #Redundacy flags --> dedundant always w2
                        new_node.redundant = True
                        new_node.redundant_id = n.id
                        n.redundant_id = new_node.id
                        #Add new element to w2.nodes
                        w2.nodes+=[new_node]
                        new_nodes_wi.append(new_node)
            elif len(nodes_corresp_w)<len(nodes_corresp_w2):
                for n, n_coord in zip(nodes_corresp_w2, nodes_corresp_w2_coord3d):
                    if n.redundant_id==-1:
                        new_node = node()
                        new_node.type = "3d" #as they will be surely conected to other facades
                        new_node.contour = True 
                        new_node.coord = n_coord[dir_w_id]
                        new_node.c = n_coord[dir_w_id]
                        if new_node.c[0]==min_x_w:
                            new_node.contour_type = "l"
                        else:
                            new_node.contour_type = "r"
                        new_node.id = new_node_element_id
                        #Update node ids for new node
                        new_node_element_id+=1  
                        #Redundacy flags --> dedundant always w2
                        n.redundant = True
                        new_node.redundant_id = n.id
                        n.redundant_id = new_node.id
                        #Add new element to w2.nodes
                        w.nodes+=[new_node]
                        new_nodes_wi.append(new_node)
            
            #If both walls have openings, then the new nodes shoud be connected to the closest one with rigid node 
            if len(w.openings)==0 or len(w2.openings)==0: #if either of walls has not openings, then need to split the piers??
                #Split elements in the wall that more points were added. Looping thorugh the wall elements, check if the element has nodes that
                #are correspondences. if so, check if the nodes are interrupted or node by the new nodes added (new node in between element nodes). If so,
                #element need to be split in n+1 elements being n the number of new nodes. The element split has to activate its split flag.
                if len(nodes_corresp_w)>len(nodes_corresp_w2): #Elements in the w2 need to be split
                    if len(w2.openings)==0:
                        #w2 nodes coorespondences including new ones
                        nodes_corresp_w2_ = np.concatenate((nodes_corresp_w2, np.array(new_nodes_wi))) #node correspondences ids including new nodes
                        nodes_corresp_w2_id_ = [n.id for n in nodes_corresp_w2] + [n.id for n in new_nodes_wi] #node correspondences ids including new nodes
                        nodes_corresp_w2_coord_ =  np.concatenate((nodes_corresp_w2_coord, np.array([n.coord for n in new_nodes_wi])), axis=0) #node correspondences coord including new nodes
                        #Sorting nodes from bottom to top
                        ind_nodes_corresp_w2_sort = np.argsort(nodes_corresp_w2_coord_[:,1]) #ind to sort accordingly to height
                        nodes_corresp_w2_ = nodes_corresp_w2_[ind_nodes_corresp_w2_sort] #sorted nodes
                        nodes_corresp_w2_id_ = np.array([nodes_corresp_w2_id_[inc] for inc in ind_nodes_corresp_w2_sort])
                        nodes_corresp_w2_coord_ = nodes_corresp_w2_coord_[ind_nodes_corresp_w2_sort]
                        #Loop thoruhg wall elements. Check if they have as nodes the correspondences. If so, check if the element is divide and create new elements.
                        for me in w2.elements:
                            if len(me.ij_nodes)>0 and me.type=='p': 
                                if me.ij_nodes[0] in nodes_corresp_w2_id_ or me.ij_nodes[1] in nodes_corresp_w2_id_:
                                    #Check if the macro element nodes are interrumped or node for new nodes. If so, needs to be split
                                    ind_order = [np.where(nodes_corresp_w2_id_==j)[0][0] for j in me.ij_nodes if j in nodes_corresp_w2_id_]
                                    if ind_order[1] - ind_order[0]>1: #it is interrupted by at least one node
                                        me.split = True
                                        #Creating new elements
                                        for j in range(ind_order[0], ind_order[1]):
                                            new_me = macro_element()
                                            new_me.type = "p"
                                            new_me.id = new_macro_element_id                    
                                            new_me.b = me.b
                                            new_me.h = nodes_corresp_w2_coord_[:,1][j+1] - nodes_corresp_w2_coord_[:,1][j] 
                                            new_me.c = [me.c[0], 0.5*(nodes_corresp_w2_coord_[:,1][j+1] + nodes_corresp_w2_coord_[:,1][j])] 
                                            n_i = nodes_corresp_w2_id_[j]
                                            n_j = nodes_corresp_w2_id_[j+1]
                                            new_me.ij_nodes = [n_i, n_j]
                                            
                                            #Add new element to the FEM.macro_elements_list[i]
                                            w2.elements += [new_me]

                                            #update me ids for next new_me
                                            new_macro_element_id+=1
                elif len(nodes_corresp_w)<len(nodes_corresp_w2):
                    if len(w.openings)==0:
                        #w nodes coorespondences including new ones
                        nodes_corresp_w_ = np.concatenate((nodes_corresp_w, np.array(new_nodes_wi))) #node correspondences ids including new nodes
                        nodes_corresp_w_id_ = [n.id for n in nodes_corresp_w] + [n.id for n in new_nodes_wi] #node correspondences ids including new nodes
                        nodes_corresp_w_coord_ =  np.concatenate((nodes_corresp_w_coord, np.array([n.coord for n in new_nodes_wi])), axis=0) #node correspondences coord including new nodes
                        #Sorting nodes from bottom to top
                        ind_nodes_corresp_w_sort = np.argsort(nodes_corresp_w_coord_[:,1]) #ind to sort accordingly to height
                        nodes_corresp_w_ = nodes_corresp_w_[ind_nodes_corresp_w_sort] #sorted nodes
                        nodes_corresp_w_id_ = np.array([nodes_corresp_w_id_[inc] for inc in ind_nodes_corresp_w_sort])
                        nodes_corresp_w_coord_ = nodes_corresp_w_coord_[ind_nodes_corresp_w_sort]
                        #Loop thoruhg wall elements. Check if they have as nodes the correspondences. If so, check if the element is divide and create new elements.
                        for me in w.elements:
                            if len(me.ij_nodes)>0 and me.type=='p': 
                                if me.ij_nodes[0] in nodes_corresp_w_id_ or me.ij_nodes[1] in nodes_corresp_w_id_:
                                    #Check if the macro element nodes are interrumped or node for new nodes. If so, needs to be split
                                    ind_order = [np.where(nodes_corresp_w_id_==j)[0][0] for j in me.ij_nodes if j in nodes_corresp_w_id_]
                                    if ind_order[1] - ind_order[0]>1: #it is interrupted by at least one node
                                        me.split = True
                                        #Creating new elements
                                        for j in range(ind_order[0], ind_order[1]):
                                            new_me = macro_element()
                                            new_me.type = "p"
                                            new_me.id = new_macro_element_id                    
                                            new_me.b = me.b
                                            new_me.h = nodes_corresp_w_coord_[:,1][j+1] - nodes_corresp_w_coord_[:,1][j] 
                                            new_me.c = [me.c[0], 0.5*(nodes_corresp_w_coord_[:,1][j+1] + nodes_corresp_w_coord_[:,1][j])] 
                                            n_i = nodes_corresp_w_id_[j]
                                            n_j = nodes_corresp_w_id_[j+1]
                                            new_me.ij_nodes = [n_i, n_j]
                                            #Add new element to the FEM.macro_elements_list[i]
                                            w.elements += [new_me]
                                            #update me ids for next new_me
                                            new_macro_element_id+=1

    #Merge redundant nodes coordinates. They will be the mean if the point lays between max and min coordinate of node cells
    for i, w in enumerate(FEM.walls_list):
        if i<len(FEM.walls_list)-1:
            w2 = FEM.walls_list[i+1]
        else:
            w2 = FEM.walls_list[0]
        #Getting nodes and ids for walls w,w2
        nodes_w_id = np.array([n.id for n in w.nodes])
        #loop through w2 nodes that are redundant. Modify the coordinates for redundant nodes in w and w2 as the mean if they lay inside the node poly_cells
        for n_w2 in w2.nodes:
            if n_w2.redundant:
                w_ind_redundant = np.where(nodes_w_id==n_w2.redundant_id)
                n_w = w.nodes[w_ind_redundant[0][0]]            
                #Geometry node in w
                if n_w.shape=='R':
                    wn_pc_coord = np.array([p.coord for pc in n_w.cells for p in pc.inter_list])
                    wn_max_z = np.max(wn_pc_coord[:,1])
                    wn_min_z = np.min(wn_pc_coord[:,1])
                #Geometry node in w2
                if n_w2.shape=='R':
                    w2n_pc_coord = np.array([p.coord for pc in n_w2.cells for p in pc.inter_list])
                    w2n_max_z = np.max(w2n_pc_coord[:,1])
                    w2n_min_z = np.min(w2n_pc_coord[:,1])
                
                #Assign neew coordinate to the correspondences nodes
                #Initial proposal of node z coordinate as the mean. Then the max and min z values of the node in w and w2
                if n_w.shape=="R" and n_w2.shape=="R":
                    new_node_coord_z_proposals = [0.5*(n_w.coord[1]+n_w2.coord[1]), wn_max_z, w2n_max_z, wn_min_z, w2n_min_z]
                    for n_coord in new_node_coord_z_proposals: #select the proposal that lays inside the nodes
                        check1 = (wn_min_z<=n_coord<=wn_max_z)
                        check2 = (w2n_min_z<=n_coord<=w2n_max_z)
                        if check1 and check2:
                            n_w.coord = np.array([n_w.coord[0], n_coord])
                            n_w2.coord = np.array([n_w2.coord[0], n_coord])
                            break
                elif n_w.shape=="R" and n_w2.shape=="N":
                    new_node_coord_z_proposals = [0.5*(n_w.coord[1]+n_w2.coord[1]), wn_max_z, wn_min_z]
                    for n_coord in new_node_coord_z_proposals: #select the proposal that lays inside the nodes
                        check1 = (wn_min_z<=n_coord<=wn_max_z)
                        if check1:
                            n_w.coord = np.array([n_w.coord[0], n_coord])
                            n_w2.coord = np.array([n_w2.coord[0], n_coord])
                            break
                elif n_w.shape=="N" and n_w2.shape=="R":
                    new_node_coord_z_proposals = [0.5*(n_w.coord[1]+n_w2.coord[1]), w2n_max_z, w2n_min_z]
                    for n_coord in new_node_coord_z_proposals: #select the proposal that lays inside the nodes
                        check2 = (w2n_min_z<=n_coord<=w2n_max_z)
                        if check2:
                            n_w.coord = np.array([n_w.coord[0], n_coord])
                            n_w2.coord = np.array([n_w2.coord[0], n_coord])
                            break
                elif n_w.shape=="N" and n_w2.shape=="N":
                    n_coord = 0.5*(n_w.coord[1]+n_w2.coord[1])
                    n_w.coord = np.array([n_w.coord[0], n_coord])
                    n_w2.coord = np.array([n_w2.coord[0], n_coord])

    #plot and save macroelements to assess
    FEM.plot_elements_EFM(labels=True, str_node=True, str_elements=True, save=True, data_folder=data_folder, contour=False)

    #Assign to the nodes and elements its correspondent wall and getting its 2d coordinate for the element.c parameter
    for w in FEM.walls_list:
        for n in w.nodes:
            n.wall = w.id
        for e in w.elements:
            e.wall = w.id
            e.c_local = copy.deepcopy(e.c)
            if w.parallel_to==0:
                e.c_local[0]-=w.coord[1]
            elif w.parallel_to==1:
                e.c_local[0]-=w.coord[0]

    #Using updated coordinate, create tremuri geometry parameters. xright, xleft, zup, zdown. 
    for w in FEM.walls_list:
        for n in w.nodes:
            if not n.vanished:
                if n.shape=="R":
                    n_pc_coord = np.array([p.coord for pc in n.cells for p in pc.inter_list])
                    n_max_x = np.max(n_pc_coord[:,0])
                    n_min_x = np.min(n_pc_coord[:,0])
                    n_max_z = np.max(n_pc_coord[:,1])
                    n_min_z = np.min(n_pc_coord[:,1])
                    n.xl = n.coord[0] - n_min_x
                    n.xr = n_max_x - n.coord[0]
                    n.zu = n_max_z - n.coord[1]
                    n.zd = n.coord[1] - n_min_z
                #Getting local coordinates
                n.coord_local = copy.deepcopy(n.coord)
                if w.parallel_to==0:
                    n.coord_local[0]-= w.coord[1]
                elif w.parallel_to==1:
                    n.coord_local[0]-= w.coord[0]


    #Creating repartizioni
    #For each 2d node in each wall select first the two 3d closest vertically using node.c.
    for w in FEM.walls_list:
        n2d = np.array([n for n in w.nodes if n.type=="2d" and not n.vanished])
        n2d_cz = np.array([n.c[1] for n in w.nodes if n.type=="2d" and not n.vanished])
        n3d = np.array([n for n in w.nodes if n.type=="3d" and not n.vanished])
        n3d_cz = np.array([n.c[1] for n in w.nodes if n.type=="3d" and not n.vanished])
        for j, cz in enumerate(n2d_cz):
            dist_cz = np.abs(n3d_cz - cz)
            two_closest_n = np.argsort(dist_cz)[:2]
            n2d[j].repart = []
            for k in two_closest_n:
                if n3d[k].redundant:
                    n2d[j].repart+=[n3d[k].redundant_id]
                else:
                    n2d[j].repart+=[n3d[k].id]
    
    #Creating rigid elements for piers in walls without openings. It just have ij nodes ids. Nodes at the same height are linked.
    new_macro_element_id = len([e.id for w in FEM.walls_list for e in w.elements])
    for w in FEM.walls_list:
        w_nodes_ids = np.array([n.id for n in w.nodes])# if len(n.cells)>0]) 
        w_nodes = [n for n in w.nodes]# if len(n.cells)>0]
        if len(w.openings)==0:
            nodes_piers_id = np.array([e.ij_nodes for e in w.elements if e.split==False])
            nodes_piers_id_unique = np.unique(nodes_piers_id)
            nodes_piers = [w_nodes[np.where(w_nodes_ids==nid)[0][0]] for nid in nodes_piers_id_unique]
            nodes_piers_coord_x = np.array([n.coord[0] for n in nodes_piers])
            npc_min_x = np.min(nodes_piers_coord_x)
            nodes_piers_left = [nn for nn in nodes_piers if nn.coord[0]==npc_min_x]
            nodes_piers_right = [nn for nn in nodes_piers if nn.coord[0]!=npc_min_x]
            nodes_piers_left_coord_z = np.array([n.coord[1] for n in nodes_piers_left])
            nodes_piers_right_coord_z = np.array([n.coord[1] for n in nodes_piers_right])
            if len(nodes_piers_right)<=len(nodes_piers_left):
                elem_ids = np.arange(int(len(nodes_piers_right))) + new_macro_element_id + 1
                w.rigid_elements = [[elem_ids[i], [nodes_piers_right[i].id, nodes_piers_left[np.argmin(np.abs(nodes_piers_left_coord_z - nodes_piers_right_coord_z[i]))].id]] for i in range(int(len(nodes_piers_right)))]
            else:
                elem_ids = np.arange(int(len(nodes_piers_left))) + new_macro_element_id + 1
                w.rigid_elements = [[elem_ids[i], [nodes_piers_left[i].id, nodes_piers_right[np.argmin(np.abs(nodes_piers_right_coord_z - nodes_piers_left_coord_z[i]))].id]] for i in range(int(len(nodes_piers_left)))]
            new_macro_element_id+=len(elem_ids)
    #Create rigid elements for the nodes that are not assigned to anything (Those nodes created because correspondence nodes number were different)
    used_nodes = []
    full_nodes_elements = [n for w in FEM.walls_list for n in w.nodes if not n.vanished]
    full_nodes_elements_ids = np.array([n.id for w in FEM.walls_list for n in w.nodes if not n.vanished])
    full_nodes_elements_w_ids = [w.id for w in FEM.walls_list for n in w.nodes if not n.vanished]
    for w in FEM.walls_list:
        for me in w.elements:
            if not me.split:
                used_nodes+=me.ij_nodes
    used_nodes = np.unique(np.array(used_nodes))
    non_used_nodes = np.setdiff1d(full_nodes_elements_ids, used_nodes)
    new_macro_element_id+=1
    for nun in non_used_nodes:
        w_id = full_nodes_elements_w_ids[np.where(full_nodes_elements_ids == nun)[0][0]]
        nodes_w_id_coord = np.array([n.coord for n in FEM.walls_list[w_id-1].nodes if not n.vanished])
        nun_coord = full_nodes_elements[np.where(full_nodes_elements_ids == nun)[0][0]].coord
        ind_samex = np.where(np.abs(nun_coord[0]-nodes_w_id_coord[:,0])<=1e-3)
        ind_closestz = np.argsort(np.abs(nun_coord[1] - nodes_w_id_coord[ind_samex][:,1]))[1] 
        closest_node_id = FEM.walls_list[w_id-1].nodes[ind_samex[0][ind_closestz]].id
        #Create rigid element for the wall
        FEM.walls_list[w_id-1].rigid_elements += [[new_macro_element_id, [nun, closest_node_id]]]
        new_macro_element_id+=1

    #Creating the boundary condition. Nodes at height z=0 non redundant are restricted
    for w in FEM.walls_list:
        for n in w.nodes:
            if n.vanished==False:
                if n.coord[1]==0:
                    n.restricted = True
    
    
    # Creating txt file with tremuri geometry
    macro_element_results_folder = '../results/'+data_folder + '/EFM_analysis'
    check_dir = os.path.isdir(macro_element_results_folder)
    if not check_dir:
        os.mkdir(macro_element_results_folder)
    tremuri_file_name = macro_element_results_folder +'/tremuri.txt'
    f = open(tremuri_file_name, 'w')    
    #HEADER
    header_lines = ['TREMURI 1 7 0\n', '/impostazioni\n', 'Default\n', 'convergenza 3\n\n\n']
    f.writelines(header_lines)
    #GEOMETRY
    #walls
    walls_lines = ['/pareti\n', "!num       X0       Y0       angle° (prefered rads)\n"]
    walls_lines += ["{}       {:.4f}       {:.4f}       {}°\n".format(w.id, w.coord[0], w.coord[1], w.angle) for w in FEM.walls_list]
    walls_lines += ["\n\n"]
    f.writelines(walls_lines)
    #materials
    materials_lines = ['/Materiali\n', "!num             E                 G           rho             fm          tau0/fvm0     fvlim verifica  driftT  driftPF     mu      Gc       beta\n"]
    materials_lines += [' 1           {:.4e}       {:.4e}       {}          {}       {}       {}        {}        {}        {}      {}     {}        {}\n'.format(E, G, rho, fm, tau_fvm, fvlim, verif, driftT, driftPF, mu, Gc, beta_pier)]
    materials_lines += ['99           {:.4e}       {:.4e}       {}          {}       {}       {}        {}        {}        {}      {}     {}        {}\n'.format(E, G, rho, fm, tau_fvm, fvlim, verif, driftT, driftPF, mu, Gc, beta_spandrel)]
    materials_lines += [' 7           {:.4e}       {:.4e}       {}\n'.format(1e20,0,0) ]
    materials_lines += ["\n\n"]
    f.writelines(materials_lines)
    #Nodes2d
    nodes2d_lines= ['/nodi2d\n', "!num  n_wall   X_local        Z         N/P/R     rho     thick     xleft      xright       zup        zdown \n"]
    nodes2d_lines+= ["{:>2d}     {}      {:.4f}      {:.4f}      {}      {}      {}      {:.4f}      {:.4f}      {:.4f}     {:.4f}\n".format(n.id, w.id, n.coord_local[0], n.coord_local[1], n.shape, rho, w_th, n.xl, n.xr, n.zu, n.zd) for w in FEM.walls_list for n in w.nodes if n.type=='2d' and n.redundant==False and n.vanished==False]
    nodes2d_lines+= ["\n\n"]
    f.writelines(nodes2d_lines)
    #Nodes3d
    nodes3d_lines= ['/nodi3d\n', "!num   n_subwalls    wall_i    wall_j    Z        N/P/R    rho    thick   xleft    xright     zup       zdown    N/P/R     rho   thick    xleft,   xright,     zup,    zdown\n"]
    w_nodes = [n for w in FEM.walls_list for n in w.nodes if not n.vanished]
    w_nodes_ids = np.array([n.id for w in FEM.walls_list for n in w.nodes if not n.vanished])
    for w in FEM.walls_list:
        for i, n in enumerate(w.nodes):
            if n.type=="3d" and n.redundant==False and n.vanished==False:
                nr = w_nodes[np.where(w_nodes_ids==n.redundant_id)[0][0]]
                if n.shape=="R" and nr.shape=="R":
                    nodes3d_lines += ["{:>2d}         {}          {}         {}       {:.4f}      {}     {}    {}    {:.4f}    {:.4f}    {:.4f}    {:.4f}      {}       {}    {}    {:.4f}    {:.4f}    {:.4f}    {:.4f}\n".format(n.id, 2, n.wall, nr.wall,  n.coord[1], n.shape, rho, w_th, n.xl, n.xr, n.zu, n.zd, nr.shape, rho, w_th, nr.xl, nr.xr, nr.zu, nr.zd)]
                elif n.shape=="R" and nr.shape=="N":
                    nodes3d_lines += ["{:>2d}         {}          {}         {}       {:.4f}      {}     {}    {}    {:.4f}    {:.4f}    {:.4f}    {:.4f}      {}\n".format(n.id, 2, n.wall, nr.wall,  n.coord[1], n.shape, rho, w_th, n.xl, n.xr, n.zu, n.zd, nr.shape)]
                elif n.shape=="N" and nr.shape=="R":
                    nodes3d_lines += ["{:>2d}         {}          {}         {}       {:.4f}      {}                                                              {}       {}    {}    {:.4f}    {:.4f}    {:.4f}    {:.4f}\n".format(n.id, 2, n.wall, nr.wall,  n.coord[1], n.shape, nr.shape, rho, w_th, nr.xl, nr.xr, nr.zu, nr.zd)]
                elif n.shape=="N" and nr.shape=="N":
                    nodes3d_lines += ["{:>2d}         {}          {}         {}       {:.4f}      {}                                                              {}\n".format(n.id, 2, n.wall, nr.wall,  n.coord[1], n.shape, nr.shape)]

    nodes3d_lines+= ["\n\n"]
    f.writelines(nodes3d_lines)
    #Floors 
    floors_lines= ['/solaio\n', "!num    nd_i    nd_j    nd_k    nd_l    thick   E1         E2              ni      G      orientation    offset    warping\n"]
    floors_lines+= ["\n\n"]
    f.writelines(floors_lines)
    #macro-elements
    macro_elements_lines = ['/elementi\n', "!num  parete incI  incJ    XBARloc      ZBAR         b          h      thick      mat    type (0=pier 1=spandrel 2=Generic+angle)\n"]
    for w in FEM.walls_list:
        for e in w.elements:
            if e.split==False and e.type=="p":
                if not w_nodes[np.where(w_nodes_ids==e.ij_nodes[0])[0][0]].redundant:
                    e_i = e.ij_nodes[0]
                else:
                    e_i = w_nodes[np.where(w_nodes_ids==e.ij_nodes[0])[0][0]].redundant_id
                if not w_nodes[np.where(w_nodes_ids==e.ij_nodes[1])[0][0]].redundant:
                    e_j = e.ij_nodes[1]
                else:
                    e_j = w_nodes[np.where(w_nodes_ids==e.ij_nodes[1])[0][0]].redundant_id  
                macro_elements_lines += ["{:>2d}     {}     {:>2d}     {:>2d}      {:.4f}      {:.4f}     {:.4f}     {:.4f}     {}       {:>2d}      {}\n".format(e.id, e.wall, e_i, e_j, e.c_local[0], e.c_local[1], e.b, e.h, w_th, 1, 0)]
        for e in w.elements: #two loops to print first piers and then spandrels
            if e.split==False and e.type=="s":
                if not w_nodes[np.where(w_nodes_ids==e.ij_nodes[0])[0][0]].redundant:
                    e_i = e.ij_nodes[0]
                else:
                    e_i = w_nodes[np.where(w_nodes_ids==e.ij_nodes[0])[0][0]].redundant_id
                if not w_nodes[np.where(w_nodes_ids==e.ij_nodes[1])[0][0]].redundant:
                    e_j = e.ij_nodes[1]
                else:
                    e_j = w_nodes[np.where(w_nodes_ids==e.ij_nodes[1])[0][0]].redundant_id     
                macro_elements_lines += ["{:>2d}     {}     {:>2d}     {:>2d}      {:.4f}      {:.4f}     {:.4f}     {:.4f}     {}       {:>2d}      {}\n".format(e.id, e.wall, e_i, e_j, e.c_local[0], e.c_local[1], e.b, e.h, w_th, 99, 1)]
    macro_elements_lines += ["\n\n"]
    f.writelines(macro_elements_lines)
    #Rigid links
    rigid_links_lines = ['/traveElastica\n', "!num wall incI   incJ mat  Area   J   InitDef  type dXi dZi dXj dZj\n"]
    for w in FEM.walls_list:
        for e in w.rigid_elements:
            if not w_nodes[np.where(w_nodes_ids==e[1][0])[0][0]].redundant:
                e_i = e[1][0]
            else:
                e_i = w_nodes[np.where(w_nodes_ids==e[1][0])[0][0]].redundant_id
            if not w_nodes[np.where(w_nodes_ids==e[1][1])[0][0]].redundant:
                e_j = e[1][1]
            else:
                e_j = w_nodes[np.where(w_nodes_ids==e[1][1])[0][0]].redundant_id         
            rigid_links_lines += ["{:>2d}    {}    {:>2d}    {:>2d}    {}    {}    {}      {}      {}   {}   {}   {}   {}\n".format(e[0], w.id, e_i, e_j, 7, 10, 5, 0, 0, 0, 0, 0, 0)]
    rigid_links_lines += ["\n\n"]
    f.writelines(rigid_links_lines)
    #mass
    f.write("/masse\n!1st floor\n\n\n")
    #planes
    f.write("/Piani    0    0\n\n\n") #!need to check if it does not need to be defined 
    #Repartition
    repartition_lines = ['/ripartizione\n', "!n2d  n3d   n3d\n"]
    repartition_lines += ["{}    {}    {}\n".format(n.id, n.repart[0], n.repart[1]) for w in FEM.walls_list for n in w.nodes if n.type=='2d']
    repartition_lines += ["\n\n"]
    f.writelines(repartition_lines)
    #Supports
    supports_lines = ['/vincoli\n', "!nodo2d     UlocX     UZ      Rot (oppure:   Nodo2P UX UY UZ RotX RotY) 1=>vincolato\n"]
    supports_lines += ["{:>2d}    v    v    v    v    v\n".format(n.id) for w in FEM.walls_list for n in w.nodes if n.restricted and n.redundant==False]
    supports_lines += ["\n\n"]
    f.writelines(supports_lines)
    f.write("!---------------------\n\n\n")
    #if modal_analysis:
    ##ANALYSIS (MODAL)
    #input
    analysis_input_lines = ["!analysis\n\n", "!modal\n/am {}\n\n\n".format(nmods), "!---------------------\n\n\n"]
    f.writelines(analysis_input_lines)
    #output
    analysis_output_lines = ["!OUTPUT\n\n", "/var period\n", "     periodo     1\n", "/var frequency\n", "     frequenza     1\n", "/var massX\n", "     ModMass.x     1\n", "/var massY\n", "     ModMass.y     1\n\n"]
    analysis_output_lines += ["/OutFileSTADATA       \"Output_modal.sta\" 	 1      	0\n", "/output   EFM_modal.txt     {:>2d}     {:>2d}\n".format(2,2), "period     {:>2d}     {:>2d}\n".format(9,3), "frequency     {:>2d}     {:>2d}\n".format(9,3), "massX     {:>2d}     {:>2d}\n".format(9,3), "massY   {:>2d}   {:>2d}\n\n".format(9,3)]
    analysis_output_lines += ["!----------------------------\n/fine"]
    f.writelines(analysis_output_lines)
    f.close()

    #Run macro element files
    if display_EFM:
        cmmd1 = ("matlab -nosplash -nodesktop -r \"tremuri_txt=\'{}/tremuri.txt\'; run(\'MAIN_1_importModel.m\');\"".format(macro_element_results_folder)) #!to visualize. Write "exit()"" to close and continue
    else:
        cmmd1 = ("matlab -nosplash -nodesktop -r \"tremuri_txt=\'{}/tremuri.txt\'; run(\'MAIN_1_importModel.m\'); exit;\"".format(macro_element_results_folder))
    os.system(cmmd1)
    if modal_analysis:
        cmmd2 = ("matlab -nodisplay -nosplash -nodesktop -r \"run(\'MAIN_2_writeAnalyses.m\'); exit;\"")
        if display_EFM:
            cmmd3 = ("matlab -nosplash -nodesktop -r \"run(\'MAIN_3_processAnalyses.m\');\"") #!to visualize - write "exit()" to close and continue
        else:
            cmmd3 = ("matlab -nosplash -nodesktop -r \"run(\'MAIN_3_processAnalyses.m\');exit;\"")
        os.system(cmmd2) 
        os.system(cmmd3) 
    #move results
    dir_list_move = ['inputFiles', 'outputFiles', 'inputFiles.mat', 'modelOpensees.mat', 'modelTremuri.mat', 'results.mat', 'gmon.out', 'EFM_batch.tcl']
    for dir_ in dir_list_move:
        if os.path.isdir(dir_) or os.path.isfile(dir_):
            shutil.move(dir_, macro_element_results_folder)
    t7 = time.time()
    print(":::Finished part (VI) in {}s:::".format(np.round(t7-t6,2)))


def FEM_main(data_folder, im1, path_builder_script, how2get_kp = 0, ctes=[.0, .3, .3, .3, .2], orientation_type = "main_normals", approach = "A", modal_analysis=False, display_EFM = False):


    # Info for generation of LOD3 model
    bound_box = 'region' 
    op_det_nn = 'unet' 
    images_path = '../data/' + data_folder + '/images/'
    cameras_path = '../data/' + data_folder + '/cameras/'
    keypoints_path = '../data/' + data_folder + '/keypoints/'
    polyfit_path = '../data/' + data_folder + '/polyfit/'
    dense = False 
    sfm_json_path = '../data/' + data_folder + '/sfm/'

    # Material properties of solid elements for modal analysis 
    E=2080e6
    nu=0.2
    rho=2360
    nmods=10
    G = 700e6
    fm = 5.44e6
    tau_fvm = 0.224e6 
    fvlim = 0
    verif = 2
    driftT = 1
    driftPF = 1
    mu = 0.15
    Gc = 5
    beta_pier = 0.5
    beta_spandrel = 0.0
    w_th = 0.30

    # Create FEM domain using solid elements
    FEM = fem_generator_solids(data_folder, dense, polyfit_path, sfm_json_path, images_path, bound_box, op_det_nn, keypoints_path, how2get_kp, im1, path_builder_script, ctes=ctes, E=E, nu=nu, rho=rho, nmods=nmods, w_th=w_th, orientation_type=orientation_type, modal_analysis=modal_analysis)

    # Create FEM geometry using macro-elements (EFM)
    FEM_generator_EFM(data_folder, approach, FEM, E, G, rho, fm, tau_fvm, fvlim, verif, driftT, driftPF, mu, Gc, beta_pier, beta_spandrel, w_th, nmods, modal_analysis = modal_analysis, display_EFM=display_EFM)