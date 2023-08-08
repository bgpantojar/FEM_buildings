import os
import numpy as np
import copy
from utils_geometry import rect_overlap, fit_plane_on_X, trimesh2edges, read_LOD, cluster_triangles2plane, proj_pt2plane, open2local, op_aligning1, op_aligning2, op_aligning3
import pymeshlab
import open3d as o3d
import open3d_tutorial as o3dtut
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import sfm
import homography
import pylab
from projection_tools import *
from shapely.geometry import Point, Polygon
import networkx as nx
from tools_obj import create_material_library
import gmsh
import shutil
from tqdm import tqdm
from shapely.ops import polygonize

class domain:
    def __init__(self) -> None:
        self.LOD2 = None
        self.openings = []
        self.LOD3 = None
        self.cracks = []
        self.planes_init = [] #planes at init coordinates. Init means place at the origin and with ground plane parallel to plane xy
        self.openings_init = []
        self.lines_LOD3 = []
        self.intersections_LOD3 = []
        self.LOD3_graph = []
        self.LOD3_poly_cells = []
        self.macro_elements_list = []
        self.nodes_list = []
        self.openings_list = []
        self.walls_list = []


    def set_init_model(self):
        #Clone planes and openings from LOD2 model 
        self.planes_init = copy.deepcopy(self.LOD2.planes)
        self.openings_init = [o for p in self.planes_init for o in p.openings]
        
        ##Update plane and opening attributes according transformations to get init conditions
        #Init transformation matrices
        T_fac_n = self.LOD2.T_fac_n
        Ts = self.LOD2.Ts
        T_init = self.LOD2.T_init
        #Updating openings
        for o in self.openings_init:
            coord_h = (np.concatenate((o.coord, np.ones((len(o.coord),1))), axis=1)).T
            o.coord = np.round((Ts @ T_init @ T_fac_n @ coord_h)[:3,:].T, 4)
        #Updating planes
        for p in self.planes_init:
            p.mesh = p.mesh.transform(Ts @ T_init @ T_fac_n)
            p.edges = p.get_edges()
            p.get_contour()
            _, current_plane_params = fit_plane_on_X(np.array(p.mesh.vertices), fit_ransac=False)
            p.params = current_plane_params
            p.get_contour_lines_ordered()
        
        #Refine meshes for planes init 
        for p in self.planes_init:
            p.mesh.remove_duplicated_vertices()
            p.mesh.remove_duplicated_triangles()
            p.mesh.remove_degenerate_triangles()
            p.mesh.remove_unreferenced_vertices()
            p.get_contour()

    
    def load_LOD2(self, dense, polyfit_path):
        #Reading LOD2 obj file and creating mesh
        #Reading LOD2 model as mesh and creating a clustered point cloud out of it
        #Loading polyfit model vertices and faces
        if dense:
            polyfit_obj_path = polyfit_path + "/polyfit_dense.obj"
        else:
            polyfit_obj_path = polyfit_path + "/polyfit.obj"

        LOD2_vertices, LOD2_triangles = read_LOD(polyfit_obj_path) 
        LOD2_mesh = o3d.geometry.TriangleMesh()
        LOD2_mesh.vertices = o3d.utility.Vector3dVector(LOD2_vertices) 
        LOD2_mesh.triangles = o3d.utility.Vector3iVector(LOD2_triangles)
        LOD2_mesh.compute_vertex_normals()
        self.LOD2 = LOD2(LOD2_mesh)
        self.LOD2.type = "LOD2"
    
    def load_LOD3(self, data_folder):
        LOD3_obj_path = "../results/" + data_folder + "/LOD3.obj"
        LOD3_vertices, LOD3_triangles = read_LOD(LOD3_obj_path) 
        LOD3_mesh = o3d.geometry.TriangleMesh()
        LOD3_mesh.vertices = o3d.utility.Vector3dVector(LOD3_vertices) 
        LOD3_mesh.triangles = o3d.utility.Vector3iVector(LOD3_triangles) 
        LOD3_mesh.compute_vertex_normals()
        self.LOD3 = LOD3(LOD3_mesh)
        self.LOD3.type = "LOD3"
    
    def assign_openings2planes(self):
        for opening in self.openings:
            self.LOD2.planes[opening.plane].openings.append(opening)
        
    def detect_redundant_openings(self, parameter = "size"):
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue
            X_openings = np.empty(shape=(0,3))
            #Read openings coordinates
            for opening in plane.openings:
                X_openings = np.concatenate((X_openings, opening.coord), axis=0)            
            #Project all opening's corners to the plane
            X_openings = proj_pt2plane(plane.params, X_openings)
            #Make it homogeneus
            X_op_hom = np.concatenate((X_openings, np.ones((len(X_openings), 1))), axis=1).T
            #Transform it to local coordinates (Z=0)
            Xl, T = open2local(X_op_hom, np.array(self.LOD2.mesh.triangle_normals), plane.params[:3]) #!
            
            #Create opening representation as rectangle [x1,y1,x2,y2] (x1,y1)bl, (x2,y2)tr
            x_openings_rectangles = []
            x_openings_rectangles_diag = []
            for ii in range(len(plane.openings)):
                X_op_rect = (Xl.T)[4*ii:4*ii+4]
                x_openings_rectangles.append([np.min(X_op_rect[:,0]), np.min(X_op_rect[:,1]), np.max(X_op_rect[:,0]), np.max(X_op_rect[:,1])])
                x_openings_rectangles_diag.append((np.min(X_op_rect[:,0])-np.max(X_op_rect[:,0]))**2 + (np.min(X_op_rect[:,1])-np.max(X_op_rect[:,1]))**2)
        
            x_openings_rectangles = np.array(x_openings_rectangles)
            x_openings_rectangles_diag = np.array(x_openings_rectangles_diag)
            #Loop to check if the rectangles overlap
            id_redundant = []
            for ii in range(len(plane.openings)):
                if ii in id_redundant:
                    continue
                id_overlap = []
                for jj in range(ii+1,len(plane.openings)):
                    if rect_overlap(x_openings_rectangles[ii], x_openings_rectangles[jj]):
                        id_overlap.append(jj)
                if len(id_overlap)>0:
                    id_overlap.append(ii)
                    id_overlap = np.array(id_overlap)
                    if parameter=="size":
                        #Select the one with bigger dimentionss
                        diag_rect = x_openings_rectangles_diag[id_overlap]
                        id_redundant += id_overlap[np.where(diag_rect!=np.max(diag_rect))[0]].tolist()
                    elif parameter=="camera":
                        #Select the closest to the camera
                        d2int_op = np.mean(np.array([plane.openings[kk].d2int for kk in id_overlap]), axis=1)
                        id_redundant += id_overlap[np.where(d2int_op!=np.min(d2int_op))[0]].tolist()
                    elif parameter=="camera-size":
                        #If the difference between sizes is too big, select the one with biggest size
                        diag_rect = x_openings_rectangles_diag[id_overlap]
                        relative_size = np.abs((diag_rect - diag_rect[0])/diag_rect[0])
                        d2int_op = np.mean(np.array([plane.openings[kk].d2int for kk in id_overlap]), axis=1)
                        if np.all(relative_size<.4): #!
                            print("relative size too small, select the biggest opening ", relative_size)
                            id_redundant += id_overlap[np.where(diag_rect!=np.max(diag_rect))[0]].tolist()
                        else:
                            print("redundant openings with different sizes, select the one with closest camera", relative_size)
                            #Select the closest to the camera
                            id_redundant += id_overlap[np.where(d2int_op!=np.min(d2int_op))[0]].tolist()
            
            #Assign redundant label to opening
            for ii in id_redundant:
                plane.openings[ii].redundant=True
                   
   
    def regularize_openings(self, ctes, use_redundant=False):
        #Loop for each plane that contain openings and regularize them using the aligning methods in local coordinates
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue
            X_openings = np.empty(shape=(0,3))
            for opening in plane.openings:
                if use_redundant:
                    X_openings = np.concatenate((X_openings, opening.coord), axis=0)
                else:
                    if not opening.redundant:
                        X_openings = np.concatenate((X_openings, opening.coord), axis=0)
            #Project all opening's corners to the plane
            X_openings = proj_pt2plane(plane.params, X_openings)
            #Make it homogeneus
            X_op_hom = np.concatenate((X_openings, np.ones((len(X_openings), 1))), axis=1).T
            #Transform it to local coordinates (Z=0)
            Xl, T = open2local(X_op_hom, np.array(self.LOD2.mesh.triangle_normals), plane.params[:3]) 
            #Aligning the width and height of the openings (Aligment 1 --> to linear regression model).
            Xl_al = op_aligning1(Xl, cte = ctes[0])      
            #CLEANING 2.1: aligning  each opening
            #Aligning the width and height of the openings (Aligment 2 --> same width and height) #!
            Xl_al2 = op_aligning2(Xl_al, cte = ctes[1])     
            #Equalizing areas
            Xl_al3 = op_aligning3(Xl_al2, cte1 = ctes[2], cte2 = ctes[3])            
            #Taking to global coordinates again
            X_al = np.dot(np.linalg.inv(T),Xl_al3)
            X_al = X_al[:3].T
            count = 0
            for ii, opening in enumerate(plane.openings):
                if use_redundant:
                    opening.coord = X_al[4*ii:4*ii+4]
                else:
                    if not opening.redundant:
                        opening.coord = X_al[4*count:4*count+4]
                        count+=1
    
    def regularize_openings_init(self, use_redundant=False):
        #Loop for each plane that contain openings and regularize
        #In this it is used the transformation matrices used in LOD2 regularization
        #in which facade planes are parallel to orthogonal world system and model placed at origin
        X_openings = np.empty(shape=(0,3))
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue            
            for opening in plane.openings:
                if use_redundant:
                    X_openings = np.concatenate((X_openings, opening.coord), axis=0)
                else:
                    if not opening.redundant:
                        X_openings = np.concatenate((X_openings, opening.coord), axis=0)
            
        #Regularizing openings in 3 orthogonal directions. Just align in the 3 different ortogonal directions the 
        #vertices with similar coordinates
        directions = [0,1,2]
        openings_v = (self.LOD2.T_fac_n @ (np.concatenate((X_openings, np.ones((len(X_openings),1))), axis=1).T))[:3,:].T
        openings_v_mod = copy.deepcopy(openings_v)
        #Thresold for regularization -- based on the volume of an axis aligned bounding box of the LOD2 model - 1% of side of cube with same volume
        aabb = self.LOD2.oriented_mesh.get_axis_aligned_bounding_box()
        threshold_reg = 0.01*aabb.volume()**(1/3) #! 
        for di in directions:
            mask_v = np.zeros(len(openings_v_mod))
            for i, v in enumerate(openings_v_mod):
                if mask_v[i] == 0:
                    id_same_coor_di = np.where(np.abs(v[di]-openings_v_mod[:,di])<threshold_reg)
                    mask_v[id_same_coor_di] = 1
                    openings_v_mod[id_same_coor_di, di] = np.mean(openings_v_mod[id_same_coor_di,di])            
        X_openings = copy.deepcopy(openings_v_mod)
        X_openings = (np.linalg.inv(self.LOD2.T_fac_n) @ (np.concatenate((X_openings, np.ones((len(X_openings),1))), axis=1).T))[:3,:].T
        

        #Updating openings coordinates
        count = 0
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue            
            for opening in plane.openings:
                if use_redundant:
                    opening.coord = X_openings[4*count:4*count+4]
                    count+=1
                else:
                    if not opening.redundant:
                        opening.coord = X_openings[4*count:4*count+4]
                        count+=1           
                           
    
    def save_openings(self, data_folder, set_init=False, move_LOD_files=False):
        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder) 
        if set_init:
            planes = self.planes_init
            file_name = "_init"
        else:
            planes = self.LOD2.planes
            file_name = ""
        cc_vv=1        
        #for plane in self.LOD2.planes:
        for plane in planes:
            if len(plane.openings)==0:
                continue
            
            #Writing an .obj file with information of the openings for each pics pair
            f = open('../results/' + data_folder + "/openings{}{}.obj".format(plane.id, file_name), "w")
            
            for opening in plane.openings:
                if not opening.redundant:
                    for X in opening.coord:
                        f.write("v {} {} {}\n".format(X[0],X[1],X[2]))
            c_v = 1 #vertices counter. Helper to identify vertices in generated faces
            num_op2create = len(plane.openings) - len([1 for op in plane.openings if op.redundant])
            for j in range(num_op2create):
                f.write("f {} {} {}\n".format(c_v,c_v+1,c_v+2))
                f.write("f {} {} {}\n".format(c_v+1,c_v+2,c_v+3))
                c_v += 4
            f.close()
    
            #Writing an .obj file with information of the openings for all of them
            f = open('../results/' + data_folder + "/openings{}.obj".format(file_name), "a")
            for opening in plane.openings:
                if not opening.redundant:
                    for X in opening.coord:
                        f.write("v {} {} {}\n".format(X[0],X[1],X[2]))
        
            for j in range(num_op2create):
                f.write("f {} {} {}\n".format(cc_vv,cc_vv+1,cc_vv+2))
                f.write("f {} {} {}\n".format(cc_vv+1,cc_vv+2,cc_vv+3))
                cc_vv += 4
            f.close()
        
        if move_LOD_files:
            #Move LOD geometry to their results folder
            LOD_results_folder = "../results/" + data_folder + "/LOD"
            check_dir = os.path.isdir(LOD_results_folder)
            if not check_dir:
                os.mkdir(LOD_results_folder)
            list_LOD_files = [ld for ld in os.listdir("../results/" + data_folder) if os.path.isfile("../results/" + data_folder + "/" + ld)]
            for f in list_LOD_files:
                shutil.move("../results/" + data_folder + "/" + f, LOD_results_folder)

    
    def save_cracks2d(self, data_folder):
        for crack in self.cracks:
            cracks2d = crack.coord2d
            np.save('../results/'+data_folder+'/cracks2d_{}.npy'.format(crack.view), cracks2d)

    def save_cracks(self, data_folder, kin_n=False, kin_t=False, kin_tn=False): 

        #Creating colormap
        if kin_n or kin_t or kin_tn:
            from matplotlib import cm
            from matplotlib.colors import ListedColormap
            top = cm.get_cmap('Oranges_r', 256)
            bottom = cm.get_cmap('Blues', 256)
            newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
            newcmp = ListedColormap(newcolors, name='OrangeBlue')

        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder) 
        cc_vv=1
        num_pts=0
        for jj, crack in enumerate(self.cracks):
            #If there is not crack kinematic information, skip
            if kin_n or kin_t or kin_tn:
                if len(crack.kinematics)==0:
                    continue

            #Creating rgb colors for saving kinematics
            if kin_n:
                cr_name = "_kin_n"
                n = crack.kinematics[:,0]
                n[np.where(n>20)] = 20 #!
                n_cmp_space = 255*(n - 0)/(20-0)
                rgb = [bottom(int(ni)) for ni in n_cmp_space]
            elif kin_t:
                cr_name = "_kin_t"
                t = crack.kinematics[:,1]
                t[np.where(t<-20)] = -20 
                t[np.where(t>20)] = 20 
                t_cmp_space = 511*(t - (-20))/(20-(-20)) 
                rgb = [newcmp(int(ti)) for ti in t_cmp_space]
            elif kin_tn:
                cr_name = "_kin_tn"
                tn = crack.kinematics[:,1]/crack.kinematics[:,0]
                tn[np.where(tn<-2)] = -2 
                tn[np.where(tn>2)] = 2 
                min_tn = np.min(tn)
                max_tn = np.max(tn)
                tn_cmp_space = 511*(tn - (-2) )/(2-(-2)) 
                rgb = [newcmp(int(tni)) for tni in tn_cmp_space]
            else:
                cr_name = ""
                rgb = [(0,0,0) for pt in crack.coord]


            #Writing an .obj file with information of the openings for each pics pair
            f = open('../results/' + data_folder + "/cracks_{}_{}{}.ply".format(jj, crack.view, cr_name), "w") #the colormap scale for this need to be modified for n and t
            f.write("ply\n\
            format ascii 1.0\n\
            element vertex {}\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property uchar red\n\
            property uchar green\n\
            property uchar blue\n\
            end_header\n".format(crack.coord.shape[0]))

            for ii, pt in enumerate(crack.coord):
                xx = np.around(pt[0],decimals=5)
                yy = np.around(pt[1],decimals=5)
                zz = np.around(pt[2],decimals=5)
                f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
            f.close()

            num_pts += len(crack.coord)
            f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "a")
            for ii, pt in enumerate(crack.coord):
                xx = np.around(pt[0],decimals=5)
                yy = np.around(pt[1],decimals=5)
                zz = np.around(pt[2],decimals=5)
                f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
            f.close()

            if jj==len(self.cracks)-1:
                f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "r")
                read_ply = f.readlines()
                read_ply.insert(0,
                "ply\n\
                format ascii 1.0\n\
                element vertex {}\n\
                property float x\n\
                property float y\n\
                property float z\n\
                property uchar red\n\
                property uchar green\n\
                property uchar blue\n\
                end_header\n".format(num_pts))
                f.close()
                
                f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "w")
                f.writelines(read_ply)
                f.close()

    def create_LOD3_lines(self):
        facade_planes_init = [p for p in self.planes_init if p.type=='f']
        id_global = 0
        for fp in facade_planes_init:
            #Find line params for
            #Facade lines
            lines_params_facade = []
            facade_contour = np.array(fp.contour.lines)
            facade_vertices = np.round(np.array(fp.mesh.vertices),4)
            dir_id = np.array(range(3))
            dir_id = np.delete(dir_id,fp.parallel_to)
            third_coord = np.round(np.mean(facade_vertices[:, fp.parallel_to]),4) #to recover later the 3D coord to the intersections 
            for l in facade_contour:
                #find line parameters using dos points over the line given by the facade vertices
                x = facade_vertices[l][:,dir_id]+1e-27 #+1e-27 to avoid singular matrices
                params_l = np.concatenate((np.linalg.inv(x) @ (-1*np.ones((2,1))), np.ones((1,1))),axis=0).reshape(-1)
                id_max_param = np.where(np.abs(params_l)==np.max(np.abs(params_l[:2])))
                params_l = params_l/params_l[id_max_param]
                lines_params_facade.append(params_l)
            lines_params_facade = np.round(np.array(lines_params_facade),4)
            #If facade produces 2 lines with the same parameters as two of its contour segments are colinear, then remove one of the lines
            lines_params_facade = np.unique(lines_params_facade, axis=0)

            #Openings lines
            op = [o for o in fp.openings if o.redundant==False]
            if len(op)>0:
                op_coord = []
                for o in op:
                    for c in o.coord:
                        op_coord.append(c)
                op_coord = np.array(op_coord)
                op_coord_2d = op_coord[:,dir_id]
                op_coord_2d_xoy = np.unique(op_coord_2d[:,0]).reshape((-1,1))
                op_coord_2d_z = np.unique(op_coord_2d[:,1]).reshape((-1,1))
                line_params_opens_vert = np.concatenate((np.ones((len(op_coord_2d_xoy), 1)), np.zeros((len(op_coord_2d_xoy), 1)), -op_coord_2d_xoy), axis=1)
                line_params_opens_hori = np.concatenate((np.zeros((len(op_coord_2d_z), 1)), np.ones((len(op_coord_2d_z), 1)), -op_coord_2d_z), axis=1)
            else:
                line_params_opens_vert = np.empty(shape=[0,3])
                line_params_opens_hori = np.empty(shape=[0,3])

            #Create line objects
            lines_params_full = [lines_params_facade, line_params_opens_vert, line_params_opens_hori]
            lines_facade_openings =  []
            id_local = 0
            for i, lines_params in enumerate(lines_params_full): 
                for lpf in lines_params:
                    line = line_fo()
                    line.plane_id = fp.id
                    line.id = id_local
                    line.id_global = id_global
                    line.params = lpf
                    line.dir_id = dir_id
                    line.third_coord = third_coord
                    if i==0:
                        line.type = "f"
                    else:
                        line.type = "o"
                    if lpf[0]==0:
                        line.dir = "h"
                    elif lpf[1]==0:
                        line.dir = "v"
                    else:
                        line.dir = "d"
                    #Append list lines on facade
                    lines_facade_openings.append(line)
                    #Update ids
                    id_local+=1
                    id_global+=1

            #append to list of FEM domain
            self.lines_LOD3.append(lines_facade_openings)

    def plot_lines_facade(self, i):
        plt.figure()
        for l in self.lines_LOD3[i]:
            if l.dir=='v':
                if l.type=='f':
                    plt.axvline(x=-l.params[2], color='b')
                else:
                    plt.axvline(x=-l.params[2], color='g')
            elif l.dir=='h':
                if l.type=='f':
                    plt.axhline(y=-l.params[2], color='b')
                else:
                    plt.axhline(y=-l.params[2], color='g')
            elif l.dir == 'd':
                if l.type=='f':
                    x = np.linspace(-2,20,10)
                    y = (-l.params[2] - l.params[0]*x)/l.params[1]
                    plt.plot(x,y, color='b')
        plt.axis('equal')
        plt.show()  

    def get_LOD3_line_intersections(self, plot=False):
        #Creating intersection objects. It is composed by the intersections that lie inside the facade and over its contours
        intersec_id_global = 0
        for lines_facade in self.lines_LOD3:
            intersections_facade = []
            intersec_id = 0
            for i, line in enumerate(lines_facade):
                for j in range(len(lines_facade)-(i+1)):
                    intersection_c = np.cross(line.params, lines_facade[i+1+j].params)
                    intersection_c = (intersection_c/intersection_c[2])[:2]
                    if (np.abs(intersection_c)==np.inf).any() or (np.isnan(np.abs(intersection_c))).any(): #check parallel lines
                        continue
                    else:
                        #recover 3D coord
                        coord_3d = np.zeros(3)
                        coord_3d[line.dir_id] = intersection_c
                        paralell_to = np.array(range(3))
                        paralell_to = np.delete(paralell_to, line.dir_id)[0]
                        coord_3d[paralell_to] = line.third_coord
                        #creating intersection class
                        intersection = inter()
                        intersection.id = intersec_id
                        intersection.id_global = intersec_id_global
                        intersection.coord = intersection_c
                        intersection.coord_3d = coord_3d
                        intersection.plane_id = line.plane_id
                        intersection.lines_id = [line.id, lines_facade[i+1+j].id]
                        intersections_facade.append(intersection)
                        intersec_id+=1
                        intersec_id_global+=1
            intersections_facade = np.array(intersections_facade)
            self.intersections_LOD3.append(intersections_facade)

        #Identify intersections that lay over facade
        facade_planes_init = [p for p in self.planes_init if p.type=='f']
        for j, fp in enumerate(facade_planes_init):
            ##ordering lines contour to make polygon with shapely
            poly = Polygon(fp.plane_vertices_ordered)
            for inte in self.intersections_LOD3[j]:
                p = Point(np.round(inte.coord,4))
                if p.within(poly):
                    inte.over_facade = True
            
            ##Select the intersections that are over the facade contour using vector rejection https://en.wikipedia.org/wiki/Vector_projection
            for inte in self.intersections_LOD3[j]:
                for ls in fp.plane_line_segments:
                    if min(ls[0],ls[2])-3e-3<=np.round(inte.coord[0],4)<=max(ls[0],ls[2])+3e-3 and min(ls[1], ls[3])-3e-3 <= np.round(inte.coord[1],4) <= max(ls[1], ls[3])+3e-3: #!th (p2_00, p4_02)
                        a = inte.coord - ls[:2]
                        b = ls[2:] - ls[:2]
                        b_ = np.array([-b[1],b[0]])
                        a2 = (a@b_)/np.linalg.norm(b)
                        if np.round(a2,2)==0:
                            inte.over_facade = True
                            inte.contour = True
            
            #The intersections that are produced by a contour line with an opening line, are inside the facade, and do not belong to the contour
            #need to be deactivated (over_facade=false). THis because those intersections are not required for the cell generation. It was 
            #a problem when the contour of the facade has a line whose extension invades the facade.
            #Intersections produced by contour lines and that are not the vertices of the facade need to be deactivated (inte.over_facade=False)
            for inte in self.intersections_LOD3[j]:
                lines_inte_id = inte.lines_id
                line0 = self.lines_LOD3[j][lines_inte_id[0]]
                line1 = self.lines_LOD3[j][lines_inte_id[1]]
                #if the intersections is inside the facade
                if inte.over_facade==True and inte.contour==False:
                    #If they are produced by contour-opening lines
                    if (line0.type=="f" and line1.type=="o") or (line0.type=="o" and line1.type=="f"):
                        inte.over_facade=False
                #if the intersecctions are produced by contour lines and are not the vertices, deactivate
                if inte.over_facade==True and inte.contour==True:
                    if line0.type=="f" and line1.type=="f":
                        distances_to_vertices = np.linalg.norm(fp.plane_vertices-inte.coord, axis=1)
                        if (distances_to_vertices>2e-3).all(): #!
                            inte.over_facade=False
                        else: # the intersection is a facade vertice -- this is useful to filter edges of the graph that are not desired
                            inte.vertex = True
            
            #Intersections that are very close each other need to be merged. eg. 3 lines intersecting in a point (in the algorithm they won't)
            #For now if the intersections are all "f"-"f"
            inte_over_facade = [inte for inte in self.intersections_LOD3[j] if inte.over_facade==True]
            coord_inte_over_facade = np.array([inte.coord for inte in inte_over_facade])
            redundant_inte = []
            full_redundant_ind = []
            for k, ci in enumerate(coord_inte_over_facade):
                redundant_inte_k = []
                for l in range(len(coord_inte_over_facade)):
                    if k+l+1 == len(coord_inte_over_facade) or k in full_redundant_ind or k+l+1 in full_redundant_ind:
                        break
                    dist_kl_ = np.linalg.norm(ci-coord_inte_over_facade[k+l+1])
                    #dist_kl.append(dist_kl_)
                    if dist_kl_<1e-3: #!
                        redundant_inte_k+=[k,k+l+1]
                if len(redundant_inte_k)>0:
                    full_redundant_ind+=redundant_inte_k
                    redundant_inte.append(list(set(redundant_inte_k)))
            if len(redundant_inte)>0:
                #Leave just one intersection with over_facade=True and increase the lines_id to 3
                for ind_k in redundant_inte:
                    inte_k = [inte_over_facade[k] for k in ind_k]
                    #check if already where checked as redundant
                    mean_coord_inte_k = np.mean(np.array([inte.coord for inte in inte_k]), axis=0)
                    lines_id_inte_k = [l_id for inte in inte_k for l_id in inte.lines_id]
                    lines_id_inte_k = list(set(lines_id_inte_k))
                    for k, inte in enumerate(inte_k):
                        if k!=0:
                            inte.over_facade = False 
                            inte.redundant = True
                        inte.coord = mean_coord_inte_k
                        inte.lines_id = lines_id_inte_k

            #letting coordinates of the intersections considerated to produce the graph
            intersections_over_facade = np.array([inte.coord for inte in self.intersections_LOD3[j] if inte.over_facade and inte.redundant==False])

            if plot:
                plt.figure()
                for l in self.lines_LOD3[j]:
                    if l.dir=='v':
                        if l.type=='f':
                            plt.axvline(x=-l.params[2], color='b')
                        else:
                            plt.axvline(x=-l.params[2], color='g')
                    elif l.dir=='h':
                        if l.type=='f':
                            plt.axhline(y=-l.params[2], color='b')
                        else:
                            plt.axhline(y=-l.params[2], color='g')
                    elif l.dir == 'd':
                        if l.type=='f':
                            x = np.linspace(-2,20,10)
                            y = (-l.params[2] - l.params[0]*x)/l.params[1]
                            plt.plot(x,y, color='b')
                plt.axis('equal')
                intersections_facade_coord = np.array([inte.coord for inte in self.intersections_LOD3[j]])
                plt.plot(intersections_facade_coord[:,0], intersections_facade_coord[:,1], 'g.')
                plt.plot(intersections_over_facade[:,0],intersections_over_facade[:,1], 'r.')
                plt.axis('equal')
                plt.show()
            
    def get_LOD3_intersections_on_lines(self):
        #Loop through intersections that lay on the facade and add the inter objects to the correspondent line
        for i, inte_facade in enumerate(self.intersections_LOD3):
            for inte in inte_facade:
                if inte.over_facade:
                    for j in inte.lines_id:
                        self.lines_LOD3[i][j].inte_list.append(inte)
    
    def plot_intersections_over_facades(self):
    #plot inside and contour intersections
        for facade_inter in self.intersections_LOD3:
            plt.figure()
            c1=0
            c2=0 
            for inte in facade_inter:
                if inte.over_facade:
                    if inte.contour:
                        plt.scatter(inte.coord[0], inte.coord[1], color = 'r')
                        c1+=1
                    else:
                        c2+=1
                        plt.scatter(inte.coord[0], inte.coord[1], color = 'g')
            plt.axis('equal')
            plt.show()
    
    def get_LOD3_graph(self):
        #Creates the graph for each building facade where nodes are intersections on the facades
        #and the edges are the line segments form together with the facade lines (facade contour and openings)
        #Defining graph nodes
        graph_nodes_LOD3_id = []
        graph_nodes_LOD3 = []
        for inte_facade in self.intersections_LOD3:
            nodes_facade_id = [inte.id for inte in inte_facade if inte.over_facade]
            nodes_facade = [inte for inte in inte_facade if inte.over_facade]
            graph_nodes_LOD3_id.append(nodes_facade_id)
            graph_nodes_LOD3.append(nodes_facade)
        
        #facade contour line segments to ignore segments that are created over facades and conect intersections "f"-"f". 
        # The segments "f"-"f" should be over the contour
        facades_planes = [fp for fp in self.planes_init if fp.type=="f"]
        
        #Defining line segments
        graph_line_segments_id = []
        graph_line_segments = []
        graph_weights = []
        for j, lines_facade in enumerate(self.lines_LOD3):
            segments_facade_id = []
            segments_facade = []
            weights_facade = []
            fp = facades_planes[j]
            for line in lines_facade:
                if len(line.inte_list)>0:
                    inte_coords = np.array([inte.coord for inte in line.inte_list])
                    inte_ids = np.array([inte.id for inte in line.inte_list])
                    #Order intersections over the line according orientation 
                    if line.dir=="h" or line.dir=="d":
                        ord_id = np.argsort(inte_coords[:,0])
                    elif line.dir=="v":
                        ord_id = np.argsort(inte_coords[:,1])
                    inte_ids_ord = inte_ids[ord_id]
                    inte_ord = [line.inte_list[i] for i in ord_id]
                    
                    #create segments for line acording to the ordered ids
                    line_segments_facade_id = [[inte_ids_ord[i], inte_ids_ord[i+1]] for i in range(len(inte_ids_ord)-1)]
                    line_segments_facade = [[inte_ord[i], inte_ord[i+1]] for i in range(len(inte_ids_ord)-1)]
                    line_weights_facade = [np.linalg.norm(inte_ord[i].coord - inte_ord[i+1].coord) for i in range(len(inte_ids_ord)-1)]

                    #Cheking if the segment is made of intersections "f"-"f". If so, they have to be part of the facade contour
                    ids_segments_to_remove = [] 
                    for k, lsf in enumerate(line_segments_facade):
                        #getting the type of lines created the intersections that make a line segment
                        lines_lsf0 = lsf[0].lines_id                        
                        line0_lsf0 = self.lines_LOD3[j][lines_lsf0[0]]
                        line1_lsf0 = self.lines_LOD3[j][lines_lsf0[1]]
                        lines_lsf1 = lsf[1].lines_id                        
                        line0_lsf1 = self.lines_LOD3[j][lines_lsf1[0]]
                        line1_lsf1 = self.lines_LOD3[j][lines_lsf1[1]]
                        #If the two inte of the segment are "f"-"f" - check if the segment is invalid (crossess facade instead of being contour segment)
                        if line0_lsf0.type=="f" and line1_lsf0.type=="f" and line0_lsf1.type=="f" and line1_lsf1.type=="f":
                            #coordinates of line segment. Check both directions
                            coord_line_segment_a = np.concatenate((lsf[0].coord, lsf[1].coord)) #possible direction a.. from 0 to 1
                            coord_line_segment_b = np.concatenate((lsf[1].coord, lsf[0].coord)) #possible direction b.. from 1 to 0
                            #distances against line segments coordinates of the facade contour
                            dist_a = np.linalg.norm(fp.plane_line_segments-coord_line_segment_a, axis=1)
                            dist_b = np.linalg.norm(fp.plane_line_segments-coord_line_segment_b, axis=1)
                            if (dist_a>3e-3).all() and (dist_b>3e-3).all(): #!
                                ids_segments_to_remove.append(k)
                    #Updating segments accordingly
                    line_segments_facade_id = [lsf for i, lsf in enumerate(line_segments_facade_id) if i not in ids_segments_to_remove]
                    line_segments_facade = [lsf for i, lsf in enumerate(line_segments_facade) if i not in ids_segments_to_remove]
                    line_weights_facade = [lwf for i, lwf in enumerate(line_weights_facade) if i not in ids_segments_to_remove]
                    #Update global LOD3 segments
                    segments_facade_id += line_segments_facade_id 
                    segments_facade += line_segments_facade
                    weights_facade += line_weights_facade
                   

            #updating graph line segments and their ids
            graph_line_segments_id.append(segments_facade_id)
            graph_line_segments.append(segments_facade)
            graph_weights.append(weights_facade)
        
        #Creating LOD3 graph
        for i in range(len(graph_nodes_LOD3_id)):
            self.LOD3_graph.append([graph_nodes_LOD3_id[i], graph_nodes_LOD3[i], graph_line_segments_id[i], graph_line_segments[i], graph_weights[i]])
    
    def plot_graph_facade(self, i=None, labels=False):

        if i is not None:
            list_iter = [i,]
        else:
            list_iter = range(len(self.LOD3_graph))

        for i in list_iter:
            plt.figure()        
            for p in self.LOD3_graph[i][1]:
                c = np.random.rand(3,)
                plt.scatter(p.coord[0], p.coord[1], color=c)
                if labels:
                    plt.annotate(str(p.id), (p.coord[0], p.coord[1]), (p.coord[0], p.coord[1]), color=c)
            for j, ls in enumerate(self.LOD3_graph[i][3]):
                c = np.random.rand(3,)
                x = [ls[0].coord[0], ls[1].coord[0]]
                y = [ls[0].coord[1], ls[1].coord[1]]
                plt.plot(x, y, color=c)
                if labels:
                    plt.annotate(str(j), (np.mean(x), np.mean(y)), (np.mean(x), np.mean(y)), color=c)
            plt.axis('equal')
            plt.show()

    def plot_cycle(self, i, cycle):
        plt.figure()
        points = [inte.coord for inte in self.LOD3_graph[i][1] if inte.id in cycle]
        points_ = points + points[0]
        lines = [[points_[j], points_[j+1]] for j in range(len(cycle))]
        for p in points:
            c=np.random.rand(3,)
            plt.scatter(p[0], p[1], color=c)
        for ls in lines:
            c = np.random.rand(3,)
            x = [ls[0][0], ls[1][0]]
            y = [ls[0][1], ls[1][1]]
            plt.plot(x, y, color=c)
        plt.axis('equal')
        plt.show()
    
    def get_LOD3_poly_cells(self, method_poly="shapely_polygonize"):

        #Defining line segments
        poly_cells_LOD3_id = []
        poly_cells_LOD3 = []
        for graph_facade in tqdm(self.LOD3_graph):
            if method_poly == "graph_cycles":
                G = nx.Graph() #create graph object from nc
                G.add_nodes_from(graph_facade[0]) #add nodes
                G.add_edges_from(graph_facade[2]) #add edges
                cycles = []
                #Get basis cycles from graph
                cycles_b = nx.cycle_basis(G,graph_facade[0][0]) #find cells of facade to be labeled as pier, spandrel or node using cycle basis starting with the node [0]
                #Get minimum basis cycles
                H = nx.Graph()
                for cy in cycles_b:
                    nx.add_cycle(H, cy)
                m_cy = nx.minimum_cycle_basis(H)
                poly_cells_facade_id =  [list(np.array(nx.find_cycle(H.subgraph(cycle)))[:, 0]) for cycle in m_cy] #recovering original order as it is not given
                poly_cells_facade = []
                poly_cell_cycle_id = 0
                for poly in poly_cells_facade_id:
                    poly_c = poly_cell()
                    poly_c.inter_list = [inte for po in poly for inte in graph_facade[1] if po==inte.id]
                    poly_c.id = poly_cell_cycle_id
                    poly_cells_facade.append(poly_c)
                    poly_cell_cycle_id+=1
            elif method_poly == "shapely_polygonize":
                line_segments = [((seg[0].coord[0], seg[0].coord[1]), (seg[1].coord[0], seg[1].coord[1])) for seg in graph_facade[3]]
                polygons = polygonize(line_segments)
                polygons_coord = [np.array(list(zip(*poly.exterior.coords))).T[:-1] for poly in polygons]
                poly_cells_facade = []
                poly_cell_cycle_id = 0
                for poly in polygons_coord:
                    poly_c = poly_cell()
                    poly_c.inter_list = [inte for po in poly for inte in graph_facade[1] if po[0]==inte.coord[0] and po[1]==inte.coord[1]]
                    poly_c.id = poly_cell_cycle_id
                    poly_cells_facade.append(poly_c)
                    poly_cell_cycle_id+=1            
            
            #Apening poly cells for LOD3
            poly_cells_LOD3.append(poly_cells_facade)
        
        #Creating LOD3 polycells [ids, inte]
        self.LOD3_poly_cells = poly_cells_LOD3
    
    def plot_LOD3_poly_cells(self, i=None):

        if i is not None:
            list_iterate = [i,]        
        else:
            list_iterate = range(len(self.LOD3_poly_cells))
        
        for i in list_iterate:            
            plt.figure()
            for cycle in self.LOD3_poly_cells[i]:
                points = [inte.coord for inte in cycle.inter_list]
                cycle_id = cycle.id
                points_array = np.array(points)
                mean_points = np.mean(points_array, axis=0)
                points_ = points + [points[0]]
                lines = [[points_[j], points_[j+1]] for j in range(len(cycle.inter_list))]
                for _, p in enumerate(points):
                    c=np.random.rand(3,)
                    plt.scatter(p[0], p[1], color=c)
                c = np.random.rand(3,)
                for ls in lines:    
                    x = [ls[0][0], ls[1][0]]
                    y = [ls[0][1], ls[1][1]]
                    plt.plot(x, y, color=c)
                plt.annotate(str(cycle_id), (mean_points[0], mean_points[1]), (mean_points[0], mean_points[1]), color=c)
            plt.axis('equal')
            plt.show()
    
    def get_neighbour_cells_ids(self):

        for i, facade_intersections in enumerate(self.intersections_LOD3):
            poly_cells =self.LOD3_poly_cells[i]
            for inte in facade_intersections:
                if inte.contour==False and inte.over_facade:
                    poly_cells_inte = [poly_cell for poly_cell in poly_cells for inte_pc in poly_cell.inter_list if inte_pc.id==inte.id]
                    mean_coord_pcs = []
                    for pc in  poly_cells_inte:
                        mean_coord_pcs.append(np.mean(np.array([i_pc.coord for i_pc in pc.inter_list]), axis=0))
                    mean_coord_pcs = np.array(mean_coord_pcs)
                    inte_vs_mean = mean_coord_pcs - inte.coord
                    #select tr,tl,br,bl among the four cells. Compare the mean coordinates against the common intersection point
                    for j, ivm in enumerate(inte_vs_mean):
                        if ivm[0]>0 and ivm[1]>0:
                            tr = j
                        elif ivm[0]<0 and ivm[1]>0:
                            tl = j
                        elif ivm[0]<0 and ivm[1]<0:
                            bl = j                
                        elif ivm[0]>0 and ivm[1]<0:
                            br = j
                    #Assigning neighbor ids to each cell
                    #for tr
                    poly_cells_inte[tr].l = poly_cells_inte[tl].id
                    poly_cells_inte[tr].bl = poly_cells_inte[bl].id
                    poly_cells_inte[tr].b = poly_cells_inte[br].id
                    #for tl
                    poly_cells_inte[tl].b = poly_cells_inte[bl].id
                    poly_cells_inte[tl].br = poly_cells_inte[br].id
                    poly_cells_inte[tl].r = poly_cells_inte[tr].id
                    #for bl
                    poly_cells_inte[bl].r = poly_cells_inte[br].id
                    poly_cells_inte[bl].tr = poly_cells_inte[tr].id
                    poly_cells_inte[bl].t = poly_cells_inte[tl].id
                    #for br
                    poly_cells_inte[br].t = poly_cells_inte[tr].id
                    poly_cells_inte[br].tl = poly_cells_inte[tl].id
                    poly_cells_inte[br].l = poly_cells_inte[bl].id
    
    def get_cells_EFM_type(self, approach="A"):
        # approach on how to discretize the facades. A: extend fist pirst horizontally and then laterally. 
        # B: extend spandrels to the bottom stopping if there are piers ate oposite sides, extend piers laterally, exten spandrels to the top. 
        # C: extend piers horizontally if there is not spandrel as neighboor, extend expandrels untila there are piers at both sides and make that cell pier.
        #To each polycell assign a label: s,p,n,o

        #restart cells labels
        for facade_poly_cells in self.LOD3_poly_cells:
            for pc in facade_poly_cells:
                pc.type=""

        facade_planes_init = [p for p in self.planes_init if p.type=='f']
        #step 1
        #assigning labels to cells that correspond to openings. Go thoruhg openings an assign the opening label to the cells that have the the mean coordinate inside the opening
        for i, fp in enumerate(facade_planes_init):
            openings_no_redundant = [o for o in fp.openings if o.redundant==False]
            if len(openings_no_redundant)>0:
                #Mean coordinates of facade cells
                facade_poly_cells = self.LOD3_poly_cells[i]
                mean_poly_cells_coord = []
                for pc in facade_poly_cells:
                    mean_poly_cells_coord.append(np.mean(np.array([i_pc.coord for i_pc in pc.inter_list]), axis=0))
                mean_poly_cells_coord = np.round(np.array(mean_poly_cells_coord),4)
                dir_id = np.array(range(3))
                dir_id = np.delete(dir_id,fp.parallel_to)
                #for each opening check the poly cell mean coord that matches with the opening centroid. Assign to the poly cell the label "o" to the type #! this did not work when openings is chopped by some of the lines
                #Check instead if the mean of the polycells lay inside the openings. Then those cells are "o"
                for op in openings_no_redundant:
                    op_coord_2d = op.coord[:,dir_id]
                    min_xy = np.min(op_coord_2d[:,0])
                    max_xy = np.max(op_coord_2d[:,0])
                    min_z = np.min(op_coord_2d[:,1])
                    max_z = np.max(op_coord_2d[:,1])
                    #check if cells mean is inside the openings
                    mask_cells_inside_openings = (mean_poly_cells_coord[:,0]>min_xy) * (mean_poly_cells_coord[:,0]<max_xy) * (mean_poly_cells_coord[:,1]>min_z) * (mean_poly_cells_coord[:,1]<max_z)
                    #Assing the label to those inside the opening
                    #facade_poly_cells[np.where(mask_cells_inside_openings)[0]].type = 'o'
                    for inside_open in np.where(mask_cells_inside_openings)[0]:
                        facade_poly_cells[inside_open].type = 'o'
                    
        #Assigning labels to the cells that are at the t,b,l,r of openings as spandrels or piers
        for facade_poly_cells in self.LOD3_poly_cells:
            #get poly_cells ids
            pcs_ids = np.array([pc.id for pc in facade_poly_cells])
            #zero pass. If there are no openings, make all a pier
            if len(facade_poly_cells)==1 or len([pc for pc in facade_poly_cells if pc.type=="o"])==0:
                facade_poly_cells[0].type = "p"
                continue
            #first pass based on the openings. First piers and spandrels. Later nodes
            #step2
            for pc in facade_poly_cells:
                if pc.type=="o":
                    #break
                    #Assign pier labels
                    if pc.l!=-1: 
                     if facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]].type = "p"
                    if pc.r!=-1:
                        if facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]].type = "p"
                    #Assign spandrel labels
                    if pc.t!=-1:
                        if facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]].type = "s"
                    if pc.b!=-1:
                        if facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]].type = "s"
            #Assign nodes labels
            #step3
            for pc in facade_poly_cells:
                if pc.type=="o":                    
                    if pc.tl!=-1: 
                        if facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type = "n"
                    if pc.bl!=-1: 
                        if facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type = "n"
                    if pc.br!=-1:
                        if facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type = "n"
                    if pc.tr!=-1: 
                        if facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type=="": facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type = "n"
            

            #!APPROACH A -> paper A
            if approach=="A":
                #Extend the piers laterally until finds a labeled cell #!
                for pc in facade_poly_cells:
                    if pc.type=="p":
                        if pc.l!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]]
                            while to_label.type=="":
                                to_label.type="p"
                                if to_label.l!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.l)[0][0]]
                                else:
                                    continue
                        if pc.r!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]]
                            while to_label.type=="":
                                to_label.type="p"
                                if to_label.r!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.r)[0][0]]
                                else:
                                    continue
                #Extend the spandrels vertically until finds a labeled cell
                for pc in facade_poly_cells:
                    if pc.type=="s":
                        if pc.t!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]]
                            while to_label.type=="":
                                to_label.type="s"
                                if to_label.t!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.t)[0][0]]
                                else:
                                    continue
                        if pc.b!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]]
                            while to_label.type=="":
                                to_label.type="s"
                                if to_label.b!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.b)[0][0]]
                                else:
                                    continue

            #!APPROACH B -- not in the paper
            elif approach=="B":
                #Extend spandrels to the bottom #!
                for pc in facade_poly_cells:
                    keep_extending = True 
                    if pc.type=="s":
                        if pc.b!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]]
                            while to_label.type=="" and keep_extending: 
                                #while to_label.type=="":
                                if not (facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="p" and facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="p"):
                                    to_label.type="s"
                                    if to_label.b!=-1:
                                        to_label = facade_poly_cells[np.where(pcs_ids==to_label.b)[0][0]]
                                    else:
                                        continue
                                else:
                                    keep_extending = False
                #Extend the piers laterally until finds a labeled cell #!
                for pc in facade_poly_cells:
                    if pc.type=="p":
                        if pc.l!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]]
                            while to_label.type=="":
                                to_label.type="p"
                                if to_label.l!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.l)[0][0]]
                                else:
                                    continue
                        if pc.r!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]]
                            while to_label.type=="":
                                to_label.type="p"
                                if to_label.r!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.r)[0][0]]
                                else:
                                    continue

                #Extend spandrels to the top
                for pc in facade_poly_cells:
                    if pc.type=="s":
                        if pc.t!=-1:
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]]
                            while to_label.type=="":
                                to_label.type="s"
                                if to_label.t!=-1:
                                    to_label = facade_poly_cells[np.where(pcs_ids==to_label.t)[0][0]]
                                else:
                                    continue

            elif approach=="C": #!paper B
                #Extend the piers laterally until finds a labeled cell. Stops if tr,br or tl,bl are spandrels
                for pc in facade_poly_cells:
                    if pc.type=="p":
                        if pc.l!=-1:
                            keep_extending = True 
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]]
                            while to_label.type=="" and keep_extending:
                                if pc.bl!=-1 and pc.tl!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="s" or facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type=="s")
                                elif pc.bl!=-1 and pc.tl==-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="s")
                                elif pc.bl==-1 and pc.tl!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type=="s")
                                if not condition:
                                    to_label.type="p"
                                    if to_label.l!=-1:
                                        to_label = facade_poly_cells[np.where(pcs_ids==to_label.l)[0][0]]
                                    else:
                                        continue
                                else:
                                    keep_extending = False
                        if pc.r!=-1:
                            keep_extending = True 
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]]
                            while to_label.type=="" and keep_extending:
                                if pc.br!=-1 and pc.tr!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="s" or facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type=="s")
                                elif pc.br!=-1 and pc.tr==-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="s")
                                elif pc.br==-1 and pc.tr!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type=="s")
                                if not condition:
                                    to_label.type="p"
                                    if to_label.r!=-1:
                                        to_label = facade_poly_cells[np.where(pcs_ids==to_label.r)[0][0]]
                                    else:
                                        continue
                                else:
                                    keep_extending = False
                
                #Extend spandrels to the bottom #!
                for pc in facade_poly_cells:
                    if pc.type=="s":
                        if pc.b!=-1:
                            keep_extending = True 
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]]
                            while to_label.type=="" and keep_extending: 
                                if pc.bl!=-1 and pc.br!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="p" and facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="p")
                                elif pc.bl!=-1 and pc.br==-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.bl)[0][0]].type=="p")
                                elif pc.bl==-1 and pc.br!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.br)[0][0]].type=="p")
                                if not condition:
                                    to_label.type="s"
                                    if to_label.b!=-1:
                                        to_label = facade_poly_cells[np.where(pcs_ids==to_label.b)[0][0]]
                                    else:
                                        continue
                                else:
                                    to_label.type="p"
                                    keep_extending = False
                        
                        if pc.t!=-1:
                            keep_extending = True #!
                            to_label = facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]]
                            while to_label.type=="" and keep_extending: 
                                if pc.tl!=-1 and pc.tr!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type=="p" and facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type=="p")
                                elif pc.tl!=-1 and pc.tr==-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.tl)[0][0]].type=="p")
                                elif pc.tl==-1 and pc.tr!=-1:
                                    condition = (facade_poly_cells[np.where(pcs_ids==pc.tr)[0][0]].type=="p")
                                if not condition:
                                    to_label.type="s"
                                    if to_label.t!=-1:
                                        to_label = facade_poly_cells[np.where(pcs_ids==to_label.t)[0][0]]
                                    else:
                                        continue
                                else:
                                    to_label.type="p"
                                    keep_extending = False

            
            #Extend the nodes laterally until finds a labeled cell #!
            for pc in facade_poly_cells:
                if pc.type=="n":
                    if pc.l!=-1:
                        to_label = facade_poly_cells[np.where(pcs_ids==pc.l)[0][0]]
                        while to_label.type=="":
                            to_label.type="n"
                            if to_label.l!=-1:
                                to_label = facade_poly_cells[np.where(pcs_ids==to_label.l)[0][0]]
                            else:
                                continue
                    if pc.r!=-1:
                        to_label = facade_poly_cells[np.where(pcs_ids==pc.r)[0][0]]
                        while to_label.type=="":
                            to_label.type="n"
                            if to_label.r!=-1:
                                to_label = facade_poly_cells[np.where(pcs_ids==to_label.r)[0][0]]
                            else:
                                continue
            
            #Extend the nodes vertically until finds a labeled cell #!
            for pc in facade_poly_cells:
                if pc.type=="n":
                    if pc.t!=-1:
                        to_label = facade_poly_cells[np.where(pcs_ids==pc.t)[0][0]]
                        while to_label.type=="":
                            to_label.type="n"
                            if to_label.t!=-1:
                                to_label = facade_poly_cells[np.where(pcs_ids==to_label.t)[0][0]]
                            else:
                                continue
                    if pc.b!=-1:
                        to_label = facade_poly_cells[np.where(pcs_ids==pc.b)[0][0]]
                        while to_label.type=="":
                            to_label.type="n"
                            if to_label.b!=-1:
                                to_label = facade_poly_cells[np.where(pcs_ids==to_label.b)[0][0]]
                            else:
                                continue


    def plot_cells_EFM(self, data_folder="", filled=True, save=False, name="", contour=True):
        #plot the facades in 2D assigning the EFM classess to the poly cells
        for i, facade_poly_cells in enumerate(self.LOD3_poly_cells):
            ec="k"
            plt.rcParams['font.size'] = '32'
            fig, ax = plt.subplots(1, figsize=(24,18))
            for pc in facade_poly_cells:
                pc_coord = np.array([inte.coord for inte in pc.inter_list])
                if pc.type=="o":
                    plt.fill(pc_coord[:,0], pc_coord[:,1], "k", edgecolor="k")
                elif pc.type=="p":
                    if filled:
                        if not contour: ec= np.array([.2,.2,1.])
                        plt.fill(pc_coord[:,0], pc_coord[:,1], color=np.array([.2,.2,1.]) , edgecolor=ec)
                    else:
                        plt.fill(pc_coord[:,0], pc_coord[:,1], "white", edgecolor=ec)
                elif pc.type=="s":
                    if filled:
                        if not contour: ec=np.array([.2,1.,.2]) 
                        plt.fill(pc_coord[:,0], pc_coord[:,1], color=np.array([.2,1.,.2]) , edgecolor=ec)
                    else:
                        plt.fill(pc_coord[:,0], pc_coord[:,1], "white", edgecolor=ec)
                elif pc.type=="n":
                    if filled:
                        if not contour: ec=np.array([.6,.6,.6])
                        plt.fill(pc_coord[:,0], pc_coord[:,1], color=np.array([.6,.6,.6]), edgecolor=ec)
                    else:
                        plt.fill(pc_coord[:,0], pc_coord[:,1], "white", edgecolor=ec)
                else:
                    if filled:
                        if not contour: ec="white" 
                        plt.fill(pc_coord[:,0], pc_coord[:,1], "white", edgecolor=ec)
                    else:
                        plt.fill(pc_coord[:,0], pc_coord[:,1], "white", edgecolor=ec)
            plt.axis("equal")
            plt.tight_layout()
            if save:
                EFM_discretization_dir = "../results/" + data_folder + "/EFM_discretization"
                check_dir = os.path.isdir(EFM_discretization_dir)
                if not check_dir:
                    os.mkdir(EFM_discretization_dir)
                fig.savefig(EFM_discretization_dir +"/"+ name +'polygonal_cells_{}.png'.format(i), bbox_inches='tight', pad_inches=0)
                fig.savefig(EFM_discretization_dir +"/" + name + 'polygonal_cells_{}.pdf'.format(i), bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                plt.show()
    
    def gen_obj_cells_EFM(self, data_folder):
        cell_types = ['o', 'p', 's', 'n',""]

        EFM_discretization_dir = "../results/" + data_folder + "/EFM_discretization"
        check_dir = os.path.isdir(EFM_discretization_dir)
        if not check_dir:
            os.mkdir(EFM_discretization_dir)

        for cell_type in cell_types:

            #Creating list of all the cells vertices that belong to the cell_type
            Poly_Vert = []
            for facade_poly_cells in self.LOD3_poly_cells:
                for pc in facade_poly_cells:
                    poly_vert = [inte.coord_3d for inte in pc.inter_list if pc.type==cell_type]
                    if len(poly_vert)>0:
                        Poly_Vert.append(poly_vert)
            
            if len(Poly_Vert)>0:
                f = open(EFM_discretization_dir + "/EFM_cells_{}.obj".format(cell_type), "w")
                f.write("mtllib {}.mtl\n".format(cell_type))
                f.write("usemtl {}\n".format(cell_type))
                check = os.path.isfile(EFM_discretization_dir + "/{}.mtl".format(cell_type))
                if not check:
                    create_material_library(EFM_discretization_dir + "/", type=cell_type)       
                for PV in Poly_Vert:
                    for V in PV:
                        f.write("v {} {} {}\n".format(V[0],V[1],V[2]))
                v_c = 1
                for pv in Poly_Vert:
                    f.write('f '+ ' '.join(str(v_c+i) for i in range(len(pv)))+'\n')
                    v_c+=len(pv)
                f.close()
        
        #Merge all the meshes in one
        ms = pymeshlab.MeshSet()
        for ct in cell_types:
            path2mesh = EFM_discretization_dir +"/EFM_cells_{}.obj".format(ct)
            check_file = os.path.isfile(path2mesh)
            if check_file:
                ms.load_new_mesh(path2mesh)
        ms.generate_by_merging_visible_meshes()
        ms.save_current_mesh(EFM_discretization_dir + '/EFM_discretization.obj')
    
    def get_macro_elements_cells(self):
        #Finding macroelemens' cells and LOD3 macro_elements and nodes lists
        macro_element_id = 1
        node_id = 1
        opening_id = 1
        for facade_poly_cells in self.LOD3_poly_cells:
            #get poly_cells ids
            #Go through the cells in each facade. Check the neighboors. If they are the same type, assign same element label. Pass to one of the neighboors
            #and check the neighboors again taking out those that have already id
            macro_elements_list_f = []
            nodes_list_f = []
            openings_list_f = []
            for pc in facade_poly_cells:
                #neighbours
                if pc.macro_element_id == -1:
                    if pc.type in ['s','p']: 
                        element_id = macro_element_id
                        element = macro_element()
                        element.type = pc.type
                    elif pc.type=='n': 
                        element_id = node_id
                        element = node()
                    elif pc.type=='o':
                        element_id = opening_id
                        element = opening()

                    pc.macro_element_id = element_id
                    nb_cells = np.array([pc.t, pc.b, pc.r, pc.l, pc.tl, pc.tr, pc.bl, pc.br]) #Maybe t,b,l,r suffices
                    nb_cells = nb_cells[np.where(nb_cells!=-1)]
                    nb_cells_same_type = [fpc for fpc in np.array(facade_poly_cells)[nb_cells] if fpc.type==pc.type and fpc.macro_element_id==-1]
                    element_cells = [pc] + nb_cells_same_type
                    while len(nb_cells_same_type)>0:
                        nb_cells = []
                        for c in nb_cells_same_type: 
                            c.macro_element_id = element_id
                            nb_cells+=[c.t, c.b, c.r, c.l, c.tl, c.tr, c.bl, c.br]
                        nb_cells = np.array(nb_cells)
                        nb_cells = nb_cells[np.where(nb_cells!=-1)]
                        nb_cells = np.unique(nb_cells)
                        nb_cells_same_type = [fpc for fpc in np.array(facade_poly_cells)[nb_cells] if fpc.type==pc.type and fpc.macro_element_id==-1]
                        element_cells += nb_cells_same_type
                    if pc.type in ['p','s']: 
                        element.id = macro_element_id
                        element.cells = element_cells
                        macro_elements_list_f.append(element)
                        macro_element_id+=1
                    elif pc.type=='n': 
                        element.id = node_id
                        element.cells = element_cells
                        element.shape = "R"
                        nodes_list_f.append(element)
                        node_id+=1
                    elif pc.type=='o':
                        element.id = opening_id
                        element.cells = element_cells
                        openings_list_f.append(element)
                        opening_id+=1

        
            #Add macro elements and nodes to domain
            self.macro_elements_list.append(macro_elements_list_f)
            self.nodes_list.append(nodes_list_f)
            self.openings_list.append(openings_list_f)

        #All the macro-elements cells should form a rectangle. Change label to the nodes that are inside
        #a bonding box that surrounded the macroelement.
        #for facade_poly_cells, facade_macro_elements, facade_nodes in zip(self.LOD3_poly_cells, self.macro_elements_list, self.nodes_list):
        for jj in range(len(self.LOD3_poly_cells)):
            facade_poly_cells = self.LOD3_poly_cells[jj]
            facade_macro_elements =  self.macro_elements_list[jj] 
            facade_macro_elements_ids =  np.array([e.id for e in facade_macro_elements]) #!
            facade_nodes = self.nodes_list[jj]
            facade_nodes_ids = np.array([n.id for n in facade_nodes])
            for me in facade_macro_elements:
                pc_ids = np.array([pc.id for pc in me.cells])
                if len(pc_ids)>0:
                    pc_coord = np.array([p.coord for pc in me.cells for p in pc.inter_list])
                    pc_ids_neighbors = np.array([[pc.t, pc.b, pc.l, pc.r] for pc in me.cells]).reshape(-1)
                    pc_ids_neighbors = np.unique(pc_ids_neighbors[np.where(pc_ids_neighbors!=-1)])
                    min_x = np.min(pc_coord[:,0])
                    max_x = np.max(pc_coord[:,0])
                    min_y = np.min(pc_coord[:,1])
                    max_y = np.max(pc_coord[:,1])
                    #Change the type for nodes that are inside the bbx defined by min max x,y
                    for pc_id in pc_ids_neighbors:
                        if facade_poly_cells[pc_id].type not in [me.type, 'o']: #!
                            cell_coord = np.mean(np.array([p.coord for p in facade_poly_cells[pc_id].inter_list]), axis=0)
                            if (min_x < cell_coord[0] < max_x) and (min_y < cell_coord[1] < max_y):
                                if facade_poly_cells[pc_id].type=="n":
                                    #Update node that contain the poly cell that will be modified
                                    node_element = facade_nodes[np.where(facade_nodes_ids==facade_poly_cells[pc_id].macro_element_id)[0][0]]
                                    node_element.cells = [c for c in node_element.cells if c.id!=pc_id]
                                    #If node_element does not have anymore cells, ten delete it 
                                    if len(node_element.cells)==0:
                                        node_element.vanished = True
                                else:
                                    element = facade_macro_elements[np.where(facade_macro_elements_ids==facade_poly_cells[pc_id].macro_element_id)[0][0]]
                                    element.cells = [c for c in element.cells if c.id!=pc_id]
                                #Change poly cell to be p or s (according to me)
                                facade_poly_cells[pc_id].type = me.type
                                facade_poly_cells[pc_id].macro_element_id = me.id
                                me.cells += [facade_poly_cells[pc_id]]

    def plot_elements_EFM(self, random_color = False, labels=False, label_contour_nodes=False, label_contour_elements=False, labels_node_type = False, str_node=False, str_elements=False, save=False, data_folder="", name="", contour=True):

        #Get full nodes ids
        full_nodes_ids = []
        full_nodes = []
        for nodes_list_f in self.nodes_list:
            full_nodes+=nodes_list_f
            full_nodes_ids+=[n.id for n in nodes_list_f]
        full_nodes_ids = np.array(full_nodes_ids)
        ec = "k" #edge color
        #plot the facades in 2D assigning the EFM classess to the poly cells
        for i in range(len(self.macro_elements_list)):   
            #Ploting macro elements     
            macro_elements_list_f = self.macro_elements_list[i]
            plt.rcParams['font.size'] = '12'
            fig, ax = plt.subplots(1, figsize=(24,18))
            for element in macro_elements_list_f:
                if len(element.cells)==0: continue
                if random_color:
                    c = np.random.rand(3,)
                else:
                    if element.type=="p":
                        c = np.array([.2,.2,1.])
                    else:
                        c = np.array([.2,1.,.2])
                
                if len(element.cells)>0:
                    pc_coord_mean = []
                    for pc in element.cells:
                        pc_coord = np.array([inte.coord for inte in pc.inter_list])
                        pc_coord_mean.append(np.mean(pc_coord,axis=0))
                        if not contour: ec=c
                        plt.fill(pc_coord[:,0], pc_coord[:,1], color=c, edgecolor=ec)
                    pc_coord_mean = np.mean(np.array(pc_coord_mean),axis=0)
                else:
                    pc_coord_mean = element.c
                if labels:
                    if element.split==True:
                        cc = 'y'
                    else:
                        cc = 'k'

                    cn = ""
                    if label_contour_elements:
                        if element.contour:
                            cn = element.contour_type+"C-"                    

                    plt.annotate(cn+element.type+str(element.id), (pc_coord_mean[0], pc_coord_mean[1]), (pc_coord_mean[0], pc_coord_mean[1]), color=cc)
                #Plot wireframes
                if str_elements==True:
                    if element.split==False: #Avoid the elements that were split
                        ni = full_nodes[np.where(full_nodes_ids==element.ij_nodes[0])[0][0]]
                        nj = full_nodes[np.where(full_nodes_ids==element.ij_nodes[1])[0][0]]
                        verts = np.array([ni.coord, element.c, nj.coord])
                        plt.plot(verts[:,0], verts[:,1], 'x--', color=np.array([1.,.2,.2]))
            #Plotting nodes
            nodes_list_f = self.nodes_list[i]
            for element in nodes_list_f:
                #if len(element.cells)>0: #avoid using the nodes that used cells whose type changed
                if not element.vanished: #avoid using the nodes that used cells whose type changed
                    c = np.array([.6,.6,.6])
                    if len(element.cells)>0:
                        pc_coord_mean = []
                        for pc in element.cells:
                            pc_coord = np.array([inte.coord for inte in pc.inter_list])
                            pc_coord_mean.append(np.mean(pc_coord,axis=0))
                            plt.fill(pc_coord[:,0], pc_coord[:,1], color=c, edgecolor=c)
                        pc_coord_mean = np.mean(np.array(pc_coord_mean),axis=0)
                    else:
                        pc_coord_mean = element.c
                    if labels:
                        cn = ""
                        if label_contour_nodes:
                            if element.contour:
                                cn = element.contour_type+"C-"
                        if labels_node_type:
                            cn = element.type + "-" + cn
                        if element.coord is None:
                            plt.annotate(cn+'n'+str(element.id), (pc_coord_mean[0], pc_coord_mean[1]), (pc_coord_mean[0], pc_coord_mean[1]), color='k')
                        else:
                            plt.annotate(cn+'n'+str(element.id), (element.coord[0], element.coord[1]), (element.coord[0], element.coord[1]), color='k')
                    if str_node:
                        plt.scatter(element.coord[0], element.coord[1], color=c, edgecolors=np.array([1.,.2,.2]), s=50)
            #Plotting openings
            openings_list_f = self.openings_list[i]
            for element in openings_list_f:
                c = np.array([0.,0.,0.])
                for pc in element.cells:
                    pc_coord = np.array([inte.coord for inte in pc.inter_list])
                    plt.fill(pc_coord[:,0], pc_coord[:,1], color=c, edgecolor="k")

            plt.axis("equal")
            plt.tight_layout()
            if save:
                EFM_discretization_dir = "../results/" + data_folder + "/EFM_discretization"
                check_dir = os.path.isdir(EFM_discretization_dir)
                if not check_dir:
                    os.mkdir(EFM_discretization_dir)
                fig.savefig(EFM_discretization_dir +"/"+ name +'macro_elements_{}.png'.format(i), bbox_inches='tight', pad_inches=0)
                fig.savefig(EFM_discretization_dir +"/" + name + 'macro_elements_{}.pdf'.format(i), bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                plt.show()

    def generate_solid_FEM(self, data_folder, w_th=.3):

        # Initialize :
        gmsh.initialize()
        # Print console mesagges
        gmsh.option.setNumber("General.Terminal", 1)
        # Create a model
        gmsh.model.add("model_0")
        #def generate_3D_FEM_geometry(self)
        #Get facade planes
        fps = [p for p in self.planes_init if p.type == "f"]
        #Get ground plane
        gp = [p for p in self.planes_init if p.type == "g"][0]
        # Create point entities
        tm = 2  # mesh size
        # Wall thickness
        #Create surface for each facade
        global_id_pt = 1
        global_id_ln = 1
        global_id_curve = 1
        global_id_surface = 1
        facades_sf_ids = []
        full_openings_curves_ids = []
        full_openings_parallel_to = []
        full_openings_extrude_dir = []
        for fp in fps:
            #FACADE
            #Define plane.extrude_direction -> if mean point of facade plane vertex extruded lays inside the ground polygon => extrude direction
            dir_id = np.array(range(3))
            dir_id = np.delete(dir_id, fp.parallel_to)
            dir_id_ground = np.array(range(3))
            dir_id_ground = np.delete(dir_id_ground, gp.parallel_to)
            g_poly = Polygon(gp.plane_vertices_ordered)
            mean_pt3d = np.zeros(3)
            mean_pt3d[dir_id] = np.mean(fp.plane_vertices_ordered, axis=0)
            mean_pt3d[fp.parallel_to] = fp.third_coord + w_th 
            mean_pt = Point(mean_pt3d[dir_id_ground])
            if mean_pt.within(g_poly):
                fp.extrude_dir = +1
            else:
                fp.extrude_dir = -1
            #Facade points
            facade_pt_ids = []
            for ls in fp.plane_line_segments:
                pt_c = np.zeros(3)
                pt_c[dir_id] = ls[:2]
                pt_c[fp.parallel_to] = fp.third_coord
                pt_c = np.round(pt_c,2)
                gmsh.model.occ.addPoint(pt_c[0],  pt_c[1],  pt_c[2], tm, global_id_pt)
                facade_pt_ids.append(global_id_pt)
                global_id_pt+=1
            #Facade lines  
            facade_ln_ids = []  
            for i in range(len(fp.plane_line_segments)):
                if i<len(fp.plane_line_segments)-1:
                    gmsh.model.occ.addLine(facade_pt_ids[i], facade_pt_ids[i+1], global_id_ln)
                else:
                    gmsh.model.occ.addLine(facade_pt_ids[i], facade_pt_ids[0], global_id_ln)
                facade_ln_ids.append(global_id_ln)
                global_id_ln+=1
            # Define facade curve loop
            gmsh.model.occ.addCurveLoop(facade_ln_ids, global_id_curve)
            facade_curve_id = global_id_curve
            global_id_curve+=1

            openings_curve_ids = []
            openings_parallel2 = []
            openings_extrude_dir = []
            if len(fp.openings)>0:
                openings_non_redundant = [o for o in fp.openings if o.redundant is False]
                #OPENINGS
                openings_pt_ids = []
                for o in openings_non_redundant:
                    op_order = [0,2,3,1]
                    op_pt_ids = []
                    for i in op_order:
                        pt_c = np.round(o.coord[i],2)
                        gmsh.model.occ.addPoint(pt_c[0],  pt_c[1],  pt_c[2], tm, tag=global_id_pt)
                        op_pt_ids.append(global_id_pt)
                        global_id_pt+=1
                    openings_pt_ids.append(op_pt_ids)
                #Facade lines  
                openings_ln_ids = []
                for j in range(len(openings_non_redundant)):  
                    op_ln_ids = []
                    for i in range(4):
                        if i<3:
                            gmsh.model.occ.addLine(openings_pt_ids[j][i], openings_pt_ids[j][i+1], tag=global_id_ln)
                        else:
                            gmsh.model.occ.addLine(openings_pt_ids[j][i], openings_pt_ids[j][0], tag=global_id_ln)
                        op_ln_ids.append(global_id_ln)
                        global_id_ln+=1
                    openings_ln_ids.append(op_ln_ids)

                # Define openings curve loops
                for op_ln_ids in openings_ln_ids:
                    gmsh.model.occ.addCurveLoop(op_ln_ids, tag=global_id_curve)
                    openings_curve_ids.append(global_id_curve)
                    openings_parallel2.append(fp.parallel_to)
                    openings_extrude_dir.append(fp.extrude_dir)
                    global_id_curve+=1
                
            
            # Add surface with facade #! THIS SEEMS THAT CREATES NEW POINTS. THEN THE pts IDs can be messed up. If I check what is the last id, it might be helpful if the next iterations starts with that id
            facade_surface_id = global_id_surface
            facades_sf_ids.append(facade_surface_id)
            gmsh.model.occ.addPlaneSurface([facade_curve_id], facade_surface_id)    
            global_id_surface+=1

            if len(fp.openings) and len(openings_non_redundant)>0:
                # Add list of opennings curve list
                full_openings_curves_ids+=openings_curve_ids
                full_openings_parallel_to+=openings_parallel2
                full_openings_extrude_dir+=openings_extrude_dir


        vol_extrude = []
        for i,fp in enumerate(fps):
            # Extrude surface to generate wall
            if fp.parallel_to==0:
                vol_extrude.append(gmsh.model.occ.extrude([(2,facades_sf_ids[i])], fp.extrude_dir*w_th, 0, 0))
            elif fp.parallel_to==1:
                vol_extrude.append(gmsh.model.occ.extrude([(2,facades_sf_ids[i])], 0, fp.extrude_dir*w_th, 0))
            else:
                vol_extrude.append(gmsh.model.occ.extrude([(2,facades_sf_ids[i])], 0, 0, fp.extrude_dir*w_th))

        # Physical groups
        volumes = [v[1][1] for v in vol_extrude]
        volumes_ = [v[1] for v in vol_extrude]

        #Boolean union
        bu1 = gmsh.model.occ.fuse(volumes_[:-1], [volumes_[-1]], removeObject=True, removeTool=True)

        #Openings surfaces
        openings_sf_ids = []
        global_id_surface+=10000 #to guarantee surface ids different
        for openings_curve_ids in full_openings_curves_ids:
            # Add surface openings
            openings_surface_id = global_id_surface
            openings_sf_ids.append(openings_surface_id)
            gmsh.model.occ.addPlaneSurface([openings_curve_ids], openings_surface_id)
            global_id_surface+=1

        vol_extrude_o = []
        for i,parallel2 in enumerate(full_openings_parallel_to):
            # Extrude surface to generate wall
            if parallel2==0:
                vol_extrude_o.append(gmsh.model.occ.extrude([(2,openings_sf_ids[i])], full_openings_extrude_dir[i]*w_th*1.05, 0, 0))
            elif parallel2==1:
                vol_extrude_o.append(gmsh.model.occ.extrude([(2,openings_sf_ids[i])], 0, full_openings_extrude_dir[i]*w_th*1.05, 0))
            else:
                vol_extrude_o.append(gmsh.model.occ.extrude([(2,openings_sf_ids[i])], 0, 0, full_openings_extrude_dir[i]*w_th*1.05))

        # Physical groups
        volumes_o = [v[1][1] for v in vol_extrude_o]
        volumes_o_ = [v[1] for v in vol_extrude_o]

        #Boolean union openings
        #Boolean difference facades-openings
        bu3 = gmsh.model.occ.cut(bu1[0], volumes_o_, removeObject=True, removeTool=True)
        volumes_LOD3 = [v[1] for v in bu3[0]]
        volumes_LOD3_ = bu3[0]

        # Synchronizing CAD with model
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, volumes_LOD3) # no tag specification

        gmsh.model.mesh.generate(3)

        # Visualizations params
        gmsh.option.setNumber('Mesh.SurfaceFaces', 0)  # see element faces
        gmsh.option.setNumber('Mesh.SurfaceEdges', 0)  # see element faces
        gmsh.option.setNumber('Mesh.VolumeFaces', 1)  # see element faces
        gmsh.option.setNumber('Mesh.Points', 1)        # see element nodes

        # Save mesh
        solids_fem_folder = "../results/" + data_folder + '/FEM_solids'
        check_dir = os.path.isdir(solids_fem_folder)
        if not check_dir:
            os.mkdir(solids_fem_folder)
        filename = solids_fem_folder + '/FEM_solids.vtk'
        gmsh.write(filename)
        # Visualize with gmsh interface
        gmsh.fltk.run()
        # Finalize gmsh
        gmsh.finalize()
    
    def run_modal_analysis_solids(self, data_folder, E=30e6, nu=0.2, rho=2400, nmods=5):
        #Run modal analysis with Amaru for Julia
        outdir = "../results/"+data_folder + "/FEM_solids"
        cmmd = ("julia FEM_modal_analysis_solids.jl {}/FEM_solids.vtk {} {} {} {} {}".format(outdir, E, nu, rho*100, nmods, outdir))#!
        os.system(cmmd) 


class LOD:
    def __init__(self, LOD2_mesh) -> None:
        self.mesh = LOD2_mesh
        self.plane_clusters = None
        self.facades = None
        self.roof = None
        self.boundary = None
        self.planes = None
        self.T_pca = None
        self.parallel_to_pca = None
        self.parallel_to_fac_n = None
        self.T_fac_n = None
        self.main_directions = None
        self.mesh_pca = None
        self.mesh_pca_mod = None
        self.facade_mesh = None
        self.T_init = None
        self.Ts = None
        self.init_mesh = None
        self.init_mesh_scaled = None
        self.type = ""
    
    def get_plane_clusters(self):
        #LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area = self.mesh.cluster_connected_triangles() #!
        LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area = cluster_triangles2plane(self)
        LOD2_triangle_clusters =  np.array(LOD2_triangle_clusters)
        LOD2_cluster_n_triangles = np.array(LOD2_cluster_n_triangles)
        LOD2_cluster_area = np.array(LOD2_cluster_area)
        LOD2_cluster_plane_params = []
        LOD2_cluster_plane_meshes = []
        #Creating plane params accordint to cluster - Visualizing clusters
        for c_id in range(len(LOD2_cluster_n_triangles)):
            triangles_no_current_cluster = LOD2_triangle_clusters!=c_id
            LOD2_mesh_current_cluster = copy.deepcopy(self.mesh)
            LOD2_mesh_current_cluster.remove_triangles_by_mask(triangles_no_current_cluster)
            LOD2_mesh_current_cluster.remove_unreferenced_vertices()
            if LOD2_cluster_area[c_id]==0:
                LOD2_cluster_area[c_id] = LOD2_mesh_current_cluster.get_surface_area()
            _, current_plane_params = fit_plane_on_X(np.array(LOD2_mesh_current_cluster.vertices), fit_ransac=False)
            LOD2_cluster_plane_params.append(current_plane_params)
            plane = Plane(LOD2_mesh_current_cluster)
            plane.id = c_id
            plane.params = current_plane_params
            LOD2_cluster_plane_meshes.append(plane)
        LOD2_cluster_plane_params = np.array(LOD2_cluster_plane_params)
        
        self.plane_clusters = [LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area, LOD2_cluster_plane_params]
        self.planes = LOD2_cluster_plane_meshes
    
    def get_boundary_line_set(self, plot = True):
        edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
        ls = o3dtut.edges_to_lineset(self.mesh, edges, (0,0,1))
        if plot:
            o3d.visualization.draw_geometries([self.mesh, ls], mesh_show_back_face=True)        
        self.boundary = ls
    
    def get_planes_contour(self):
        for plane in self.planes:
            plane.get_edges()
            plane.get_contour()
        
    
    def plot_mesh(self):
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([self.mesh, coord_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
 
    def plot_boundary(self):
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([self.boundary, coord_mesh])
    
    def save_mesh(self,data_folder):
        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder)
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}.obj".format(self.type), self.mesh,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)
   
    def decimate(self):
        LOD2_mesh_decimated = o3d.geometry.TriangleMesh()
        for plane in self.planes:
            plane.decimate(automatic=True)
            LOD2_mesh_decimated+=plane.mesh
        
        LOD2_mesh_decimated.merge_close_vertices(1e-5) #!
        self.mesh = LOD2_mesh_decimated

    def check_properties(self):
        self.mesh.compute_vertex_normals()

        edge_manifold = self.mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = self.mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = self.mesh.is_vertex_manifold()
        self_intersecting = self.mesh.is_self_intersecting()
        watertight = self.mesh.is_watertight()
        orientable = self.mesh.is_orientable()

        print("LOD2 mesh properties")
        print(f"  edge_manifold:          {edge_manifold}")
        print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
        print(f"  vertex_manifold:        {vertex_manifold}")
        print(f"  self_intersecting:      {self_intersecting}")
        print(f"  watertight:             {watertight}")
        print(f"  orientable:             {orientable}")

        geoms = [self.mesh]
        if not edge_manifold:
            edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=True)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (1, 0, 0)))
        if not edge_manifold_boundary:
            edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (0, 1, 0)))
        if not vertex_manifold:
            verts = np.asarray(self.mesh.get_non_manifold_vertices())
            pcl = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices)[verts]))
            pcl.paint_uniform_color((0, 0, 1))
            geoms.append(pcl)
        if self_intersecting:
            intersecting_triangles = np.asarray(
                self.mesh.get_self_intersecting_triangles())
            intersecting_triangles = intersecting_triangles[0:1]
            intersecting_triangles = np.unique(intersecting_triangles)
            print("  # visualize self-intersecting triangles")
            triangles = np.asarray(self.mesh.triangles)[intersecting_triangles]
            edges = [
                np.vstack((triangles[:, i], triangles[:, j]))
                for i, j in [(0, 1), (1, 2), (2, 0)]
            ]
            edges = np.hstack(edges).T
            edges = o3d.utility.Vector2iVector(edges)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (1, 0, 1)))
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)
    
    def get_plane_labels(self, method='pca'):

        #3D models
        mesh = self.mesh
        pcd = mesh.sample_points_uniformly(number_of_points=500000)
        
        if method=='pca':
            X = np.array(pcd.points)
            pca = PCA(n_components=3)
            pca.fit(X)
            main_directions = pca.components_
            mean_coord = pca.mean_
        elif method=='main_normals':
            mean_coord = pcd.compute_mean_and_covariance()[0]
            planes_normals = []
            for p in self.planes:
                n = p.params[:3]
                n = n/np.linalg.norm(n)
                planes_normals.append(n)
            planes_normals = np.array(planes_normals)
            #Find the planes that contain the plane normals of the LOD model (this for each pair of normals)
            main_planes_normals = [] #list with the plane normals that contain coumple of vectors (each vector being a normal of the LOD planes)
            main_planes_normals_ind = []
            for i, normal_i in enumerate(planes_normals):
                #main_planes_normals_i = []
                for j in range(len(planes_normals)):
                    if j+i+1==len(planes_normals):
                        break
                    main_planes_normals.append(np.cross(normal_i, planes_normals[i+j+1]))
                    main_planes_normals_ind.append([i,i+j+1])
            main_planes_normals = np.array(main_planes_normals)
            #As the LOD normals are normalized, the normal of candidate planes that represent the direction of the LOD building should have a magnitude
            #close to the unity. Also there will be some repeated normals (at least rounded 1 decimal). Finally some with the same orientation but not same
            #direction that represent same planes. Next these three conditions are filtered
            main_planes_normals_unique_1, ind_unique_1 = np.unique(np.round(main_planes_normals,1), axis=0, return_index=True)
            main_planes_normals_unique_2, ind_unique_2 = np.unique(np.abs(np.round(main_planes_normals_unique_1,1)), axis=0, return_index=True)
            main_planes_normals_norm_u2 = np.linalg.norm(main_planes_normals_unique_2, axis=1)
            ind_unit_n = np.where(main_planes_normals_norm_u2>0.95)[0] #!
            candidate_main_planes_normals = main_planes_normals[ind_unique_1[ind_unique_2[ind_unit_n]]]
            candidate_main_planes_normals = np.array([cmpn/np.linalg.norm(cmpn) for cmpn in candidate_main_planes_normals])
            
            #Need to select the plane that contains most of the LOD normals. The candidate plane's normal together with 
            # LOD normals that generate it represent the 3 main directions of the LOD model. THis replaces the pca sistem that 
            # does not work properly when the building in plant does not have simetric facades. 
            # For this project the LOD normals to the candidates. The candidate that gives the most quantity of nule projections is the selected plane. 
            planes_normals_dot_candidate_main_planes = np.round(planes_normals @ candidate_main_planes_normals.T,1)
            main_plane_normal_ind = np.argmax(np.sum(planes_normals_dot_candidate_main_planes==0, axis=0))
            main_plane_normal = candidate_main_planes_normals[main_plane_normal_ind]
            other_main_dir_ind = main_planes_normals_ind[ind_unique_1[ind_unique_2[ind_unit_n]][main_plane_normal_ind]]
            main_directions = np.array([planes_normals[other_main_dir_ind[0]], planes_normals[other_main_dir_ind[1]], main_plane_normal])
        
        #Defining LOD model direction
        initial_system = np.array([[0,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1]]).T
        #lod_system = np.concatenate((pca.mean_.reshape((-1,3)), pca.components_+pca.mean_))
        lod_system = np.concatenate((mean_coord.reshape((-1,3)), main_directions +mean_coord))
        lod_system = np.concatenate((lod_system.T, np.ones((1,4))))
        T = initial_system @ np.linalg.inv(lod_system)
        if self.T_pca is None:
            self.T_pca = T

        #Selection of facades, roof, ground
        planes_dot_pca = []
        for p in self.planes:
            #break
            p_dot_pca = []
            for c in main_directions:
                n = p.params[:3]
                n = n/np.linalg.norm(n)
                p_dot_pca.append(n@c)
            planes_dot_pca.append(p_dot_pca)
        planes_dot_pca = np.abs(np.array(planes_dot_pca))
                
        parallel_to = np.zeros(len(self.planes))
        parallel_to_value = np.max(planes_dot_pca, axis=1)
        parallel_to_value_arg = np.argmax(planes_dot_pca, axis=1)
        pca_planes_scores = [np.mean(parallel_to_value[np.where(parallel_to_value_arg==i)[0]]) for i in range(3)]
        dir_ground_roof = np.argmin(pca_planes_scores)
        #Potential facades
        dir_facades = np.setdiff1d(np.array([0,1,2]), np.array([dir_ground_roof]))
        ids_initial_non_ground_roof = np.where(planes_dot_pca[:, dir_facades]>0.85)[0] #!
        ids_ground_roof = np.setdiff1d(np.arange(len(self.planes)),ids_initial_non_ground_roof)
        #planes that do not share a node with other facades, they are actually roof
        ids_extra_roof = []
        for i in ids_initial_non_ground_roof:
            coord_facade_i = np.array(self.planes[i].mesh.vertices)
            ids_others = np.setdiff1d(ids_initial_non_ground_roof, np.array([i]))
            counter_common_v = 0
            for jj in ids_others:
                coord_facade_jj = np.array(self.planes[jj].mesh.vertices)
                if len([cfi for cfi in coord_facade_i for cfjj in coord_facade_jj if np.linalg.norm(cfi-cfjj)==0])==0:
                    counter_common_v +=1
            if len(ids_others)==counter_common_v:
                ids_extra_roof.append(i)
        #Update ground_roof with roof elements htat are not facade as they are sorrounded by roof
        ids_ground_roof = (np.concatenate((ids_ground_roof, np.array(ids_extra_roof)))).astype('int')
        #Select ids for facades, roof and floor
        id_facades = np.setdiff1d(np.arange(len(self.planes)),ids_ground_roof)
        #Updating parallel to
        parallel_to[ids_ground_roof.astype('int')] = dir_ground_roof
        parallel_to[id_facades[np.where(planes_dot_pca[id_facades, dir_facades[0]]>.8)]] = dir_facades[0] #!
        parallel_to[id_facades[np.where(planes_dot_pca[id_facades, dir_facades[1]]>.8)]] = dir_facades[1] #!

        #The selected ground plane should be connected to most of facades
        coord_facades = [np.array(self.planes[i].mesh.vertices) for i  in id_facades]
        ground_roof_fac_linked = []
        for i in ids_ground_roof:
            coord_ground = np.array(self.planes[i].mesh.vertices)
            number_facades_linked = 0
            #Check if facade shares nodes with potential ground plane
            for cfs in coord_facades:
                if len([cg for cg in coord_ground for cf in cfs if np.linalg.norm(cf-cg)==0])>0:
                    number_facades_linked+=1
            ground_roof_fac_linked.append(number_facades_linked)
            #check if all the facades are sharing at least one node with ground
        ground_roof_fac_linked = np.array(ground_roof_fac_linked)
        id_ground = ids_ground_roof[np.argmax(ground_roof_fac_linked)]
                
        id_roof = ids_ground_roof[np.where(ids_ground_roof!=id_ground)]
        self.planes[id_ground].type = "g"
        for i in id_roof:
            self.planes[i].type = "r"
        for i in id_facades:
            self.planes[i].type = "f"


    def pre_regularize(self):
        """creates the transformation matrix T and parallel to arrange necessary to regularize
        the LOD model. This findes the orientation of the ground face that is perpendicular to the
        the facade faces. Based on this T is computed and then facade vertices are modified
        to guaranty ortogonallity"""

        facade_planes = [p for p in self.planes if p.type=='f']
        facade_planes_ids = np.array([p.id for p in facade_planes])
        #Cluster the facades with approx same normals. This to find a plane that represents the ground
        facade_planes_normals = np.array([p.params[:3]/np.linalg.norm(p.params[:3]) for p in facade_planes])
        
        #If pre-regularize requires computations of main directions
        if self.main_directions is None:
            facade_planes_same_normals = np.abs(np.round(facade_planes_normals @ facade_planes_normals.T,1)) #!
            facade_planes_same_normals = np.unique(facade_planes_same_normals, axis=0)
            facade_planes_same_normals_ids = [np.where(l==1)[0] for l in facade_planes_same_normals]
            #Select the most repeated direction (Maybe good to combine with the area. It must influence the bigger areas)
            most_repeated_id = np.argmax(np.array([len(fpsn) for fpsn in facade_planes_same_normals_ids]))
            most_repeated_normals = facade_planes_same_normals_ids[most_repeated_id]
            ##Mean normals most repeated orientation and its perpendicular
            mean_most_repeated = np.mean(facade_planes_normals[most_repeated_normals], axis=0)
            #Perpendicular facades normals
            most_repeated_normals_perp = np.where(facade_planes_same_normals[most_repeated_id]==0) #!
            mean_most_repeated_perp = np.mean(facade_planes_normals[most_repeated_normals_perp], axis=0)
            

            #Mean normals of facades with same orientation #!
            #Normal ground plane
            normal_ground_plane = np.cross(mean_most_repeated,mean_most_repeated_perp).reshape((-1,3))
            #Geometrical center of mesh?
            gc_LOD2 = np.mean(np.array(self.mesh.vertices), axis=0).reshape((-1,3))
            #Check if normal to ground is pointing upwards
            point_ground = ([p for p in self.planes if p.type=='g'][0].mesh.vertices)[0]
            if np.linalg.norm((gc_LOD2+normal_ground_plane)-point_ground)<np.linalg.norm((gc_LOD2-normal_ground_plane)-point_ground):
                normal_ground_plane = -normal_ground_plane
                mean_most_repeated_perp = -mean_most_repeated_perp

            #Finding transformation matrix in the way that z points as normal of the ground
            initial_system = np.array([[0,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1]]).T
            #lod_system = np.concatenate((gc_LOD2, (facades_mean_normals[0]+gc_LOD2).reshape((-1,3)), (facades_mean_normals[1]+gc_LOD2).reshape((-1,3)), normal_ground_plane + gc_LOD2))
            lod_system = np.concatenate((gc_LOD2, (mean_most_repeated+gc_LOD2).reshape((-1,3)), (mean_most_repeated_perp+gc_LOD2).reshape((-1,3)), normal_ground_plane + gc_LOD2))
            lod_system = np.concatenate((lod_system.T, np.ones((1,4))))
            T = initial_system @ np.linalg.inv(lod_system)
            self.T_fac_n = T
            self.main_directions = [mean_most_repeated, mean_most_repeated_perp, normal_ground_plane]
        else:
            [mean_most_repeated, mean_most_repeated_perp, normal_ground_plane] = self.main_directions
        #Defining the facade planes to which direccion are parallel
        parallel_to = np.ones(len(self.planes))*2 #by default all are roofs??.. parallel to xy plane
        for i, n in enumerate(facade_planes_normals):
            parallel_0 = abs(n@mean_most_repeated)
            parallel_1 = abs(n@mean_most_repeated_perp)
            if parallel_0>parallel_1:
                parallel_to[facade_planes_ids[i]] = 0
            else:
                parallel_to[facade_planes_ids[i]] = 1   
        
        self.parallel_to_fac_n = parallel_to.astype('int')

        #Update parallel to for planes #!
        for i, p in enumerate(self.planes):
            p.parallel_to = self.parallel_to_fac_n[i]
    
    def regularize_LOD(self, data_folder, type_transformation="facade_normals"):
        """
        It transforms the LOD model to the origing oriented in strategic way to
        to regulalize it.

        Args:
            type_transformation (str, optional): _description_. Defaults to "facade_normals".
        """

        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder)

        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        LOD2_copy = copy.deepcopy(self.mesh)

        if type_transformation=="pca_regularizer":
            T = self.T_pca
            parallel_to = self.parallel_to_pca
            print("Regularizing with PCA approach---")
        elif type_transformation=="facade_normals":
            print("Regularizing with facade normals approach---")
            T = self.T_fac_n
            parallel_to = self.parallel_to_fac_n
        
        #Define LOD2 transformed to origin oriented according to PCA and planes
        self.mesh_pca = copy.deepcopy(self.mesh).transform(T)
        self.mesh_pca_mod = copy.deepcopy(self.mesh).transform(T)
        for plane in self.planes:
            plane.mesh_pca = copy.deepcopy(plane.mesh).transform(T)
            plane.mesh_pca_mod = copy.deepcopy(plane.mesh).transform(T)                            
        #Modify planes to be orthogonal following pca (At least ground and facades)
        LOD2_mesh_t = self.mesh_pca_mod
        for plane in self.planes:
            #break
            if plane.type == "r":
                continue
            else:
                p_mesh_t = plane.mesh_pca_mod
                p_mesh_t_v = np.array(p_mesh_t.vertices)
                p_plane_params_modified = np.zeros(4)
                p_plane_params_modified[parallel_to[plane.id]] = 1
                if plane.type == 'g':
                    p_plane_params_modified[3] = -max(p_mesh_t_v[:, parallel_to[plane.id]].max(), p_mesh_t_v[:, parallel_to[plane.id]].min(), key=abs)
                else:
                    p_plane_params_modified[3] = -np.mean(p_mesh_t_v[:, parallel_to[plane.id]])
                #update meshes vertices (planes and LOD2)
                p_mesh_t_v_mod = copy.deepcopy(p_mesh_t_v)
                p_mesh_t_v_mod[:, parallel_to[plane.id]] = -p_plane_params_modified[3]
                p_mesh_t.vertices = o3d.utility.Vector3dVector(p_mesh_t_v_mod)
                LOD2_mesh_t_v = np.array(LOD2_mesh_t.vertices)
                #Find the ids of the vertices to be modified in the LOD2 mesh
                idx = np.array([[i,j] for i in range(len(p_mesh_t_v)) for j in range(len(LOD2_mesh_t_v)) if (np.round(LOD2_mesh_t_v[j],4)==np.round(p_mesh_t_v[i],4)).all()])
                LOD2_mesh_t_v[idx[:,1]] = p_mesh_t_v_mod[idx[:,0]]
                LOD2_mesh_t.vertices = o3d.utility.Vector3dVector(LOD2_mesh_t_v)
                for plane_ in self.planes:
                    if plane_.id == plane.id:
                        continue
                    p_mesh_t_ = plane_.mesh_pca_mod
                    p_mesh_t_v_ = np.array(p_mesh_t_.vertices)
                    #Find the ids of the vertices to be modified in the plane mesh
                    idx_ = np.array([[i,j] for i in range(len(p_mesh_t_v)) for j in range(len(p_mesh_t_v_)) if (p_mesh_t_v_[j]==p_mesh_t_v[i]).all()])
                    if len(idx_)>0:
                        p_mesh_t_v_[idx_[:,1]] = p_mesh_t_v_mod[idx_[:,0]]
                        p_mesh_t_.vertices = o3d.utility.Vector3dVector(p_mesh_t_v_)

                #update roof vertices

        #Regularizing mesh in 3 orthogonal directions. Just align in the 3 different ortogonal directions the 
        #vertices with similar coordinates
        directions = [0,1,2]
        m_mesh_t = self.mesh_pca_mod
        m_mesh_t_v = np.array(m_mesh_t.vertices)
        #update meshes vertices (planes and LOD2)
        m_mesh_t_v_mod = copy.deepcopy(m_mesh_t_v)
        #Thresold for regularization -- based on the volume of an axis aligned bounding box of the LOD2 model - 1% of side of cube with same volume
        aabb = m_mesh_t.get_axis_aligned_bounding_box()
        threshold_reg = 0.03*aabb.volume()**(1/3) #!
        for di in directions:
            mask_v = np.zeros(len(m_mesh_t_v_mod))
            for i, v in enumerate(m_mesh_t_v_mod):
                if mask_v[i] == 0:
                    id_same_coor_di = np.where(np.abs(v[di]-m_mesh_t_v_mod[:,di])<threshold_reg)
                    mask_v[id_same_coor_di] = 1
                    m_mesh_t_v_mod[id_same_coor_di, di] = np.mean(m_mesh_t_v_mod[id_same_coor_di,di])            
        m_mesh_t.vertices = o3d.utility.Vector3dVector(m_mesh_t_v_mod)
        
        #Update LOD2 mesh roof reg
        #Find the ids of the vertices to be modified in the LOD2 mesh
        #Update planes roof reg
        for plane_ in self.planes:
            p_mesh_t_ = plane_.mesh_pca_mod
            p_mesh_t_v_ = np.array(p_mesh_t_.vertices)
            #Find the ids of the vertices to be modified in the plane mesh
            idx_ = np.array([[i,j] for i in range(len(m_mesh_t_v)) for j in range(len(p_mesh_t_v_)) if (p_mesh_t_v_[j]==m_mesh_t_v[i]).all()])
            if len(idx_)>0:
                p_mesh_t_v_[idx_[:,1]] = m_mesh_t_v_mod[idx_[:,0]]
                p_mesh_t_.vertices = o3d.utility.Vector3dVector(p_mesh_t_v_)
        
        #Updating LOD2
        self.mesh = copy.deepcopy(self.mesh_pca_mod).transform(np.linalg.inv(T))
        #Remove strange obj 
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_unreferenced_vertices()
        
        #Updating planes
        for plane in self.planes:
            plane.mesh = copy.deepcopy(plane.mesh_pca_mod).transform(np.linalg.inv(T))

        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_regularized.obj".format(self.type), self.mesh,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_copy.obj".format(self.type), LOD2_copy,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)

    
    def create_facade_model(self, data_folder):
        self.facade_mesh = o3d.geometry.TriangleMesh()
        planes_facade = [p.mesh for p in self.planes if p.type=='f']
        for p in planes_facade:
            self.facade_mesh+=p
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_facade.obj".format(self.type), self.facade_mesh,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)
    
    def create_roof_model(self, data_folder):
        self.roof_mesh = o3d.geometry.TriangleMesh()
        planes_roof = [p.mesh for p in self.planes if p.type=='r']
        for p in planes_roof:
            self.roof_mesh+=p
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_roof.obj".format(self.type), self.roof_mesh,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)
    
    def create_ground_model(self, data_folder):
        self.ground_mesh = o3d.geometry.TriangleMesh()
        planes_ground = [p.mesh for p in self.planes if p.type=='g']
        for p in planes_ground:
            self.ground_mesh+=p
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_ground.obj".format(self.type), self.ground_mesh,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)


    #def get_LOD_init(self, data_folder, im1="", images_path="", intrinsic=None, poses=None, i=0, s=None):
    def get_LOD_init(self, data_folder, im1="", images_path="", intrinsic=None, poses=None, i=0):
        #Placing the LOD2 and LOD3 models oriented and at the (0,0,0) initial coordinate and scaling it acording two points selected on one image
        
        self.oriented_mesh = copy.deepcopy(self.mesh).transform(self.T_fac_n)
        min_x_coord = np.min(np.array(self.oriented_mesh.vertices)[:,0])
        min_y_coord = np.min(np.array(self.oriented_mesh.vertices)[:,1])
        min_z_coord = np.min(np.array(self.oriented_mesh.vertices)[:,2])
        T_init = np.eye(4)
        T_init[:3,3] = -np.array([min_x_coord, min_y_coord, min_z_coord])
        self.T_init = T_init
        self.init_mesh = copy.deepcopy(self.oriented_mesh).transform(T_init)
        coord_init_mesh = np.array(self.init_mesh.vertices)
        len(coord_init_mesh)

        #if self.Ts is None and s is None:
        if self.Ts is None:
            view_name = im1[i]
            #Scaling LOD model
            #Reading image to be used for scaling the building
            if os.path.isfile(images_path + "im1/" + view_name + '.jpg'):
                imm1 = np.array(Image.open(images_path + "im1/" + view_name + '.jpg'))
            elif os.path.isfile(images_path + "im1/" + view_name + '.JPG'):
                imm1 = np.array(Image.open(images_path + "im1/" + view_name + '.JPG'))
            else:
                imm1 = np.array(Image.open(images_path + "im1/" + view_name + '.png'))
            #Selecting  correspondence points to be projected from image 1
            plt.figure()
            plt.imshow(imm1)
            print('Please click {} points for scaling the LOD model'.format(2))
            x1 = np.array(pylab.ginput(2,200))
            print('you clicked:',x1)
            plt.close()
            # make homogeneous and normalize with inv(K)
            x1 = homography.make_homog(x1.T)
            #Reading camera poses information
            P, K, R, t, k_dist, P_type = camera_matrices(intrinsic, poses, return_Rt=True)
            P1_inid = P[im1[i]]['intrinsicId']
            #Correcting cordinates by lens distortion
            if P_type[P1_inid]=='radial3':
                x1_u = undistort_points(x1.T,K[P1_inid],k_dist[P1_inid]).T
            elif P_type[P1_inid]=='fisheye4':
                x1_u = undistort_points_fish(x1.T, K[P1_inid], k_dist[P1_inid]).T    
            x1 = x1_u
            R_view = R[im1[i]]
            t_view = t[im1[i]]
            #Transformation matrix to transform image plane at pose place in 3D
            T_x3D = sfm.compute_T_for_x_to_3D(R_view, t_view)
            #Finding camera center and image plane coordinates of corners and points in 3D space
            c_h = np.array([0,0,0,1]).reshape((-1,1)) #camera_center
            c_3D = T_x3D @ c_h
            c_3D /= c_3D[3]
            X = []
            #Creating list of rays passing by camera center and x1
            rays_list = []
            for v in x1.T:
                v_h = v.reshape((-1,1))
                v_h_norm = np.linalg.inv(K[poses[im1[i]]['intrinsicId']]) @ v_h        
                v_3D = T_x3D @ np.concatenate((v_h_norm, np.ones((1,1))), axis=0)
                v_3D /= v_3D[3]        
                ray_dir = v_3D[:3] - c_3D[:3]
                ray_dir = ray_dir/np.linalg.norm(ray_dir)
                rays_list.append([c_3D[0][0], c_3D[1][0], c_3D[2][0], ray_dir[0][0], ray_dir[1][0], ray_dir[2][0]])
            #Using ray casting to find 3D X coorespondences of x        
            LOD2_mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(LOD2_mesh_)
            rays = o3d.core.Tensor(rays_list, dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays)
            dist2int = (ans['t_hit'].numpy()).reshape((-1,1))
            dist2int = np.concatenate((dist2int, dist2int, dist2int), axis = 1)
            rays = rays.numpy()
            X = rays[:,:3] + dist2int*rays[:,3:]        
            if np.sum(X==np.inf)>0:
                print("One of the rays did not hit the mesh. Select other points")
            else:
                redo = False
            #Computing scaling factor s
            distance_real = float(input("please write in m the distance in real world of the points"))
            distance_3D = np.linalg.norm(X[1]-X[0])
            s = distance_real/distance_3D
            Ts = s*np.eye(4)
            Ts[3,3] = 1
            self.Ts = Ts

        self.init_mesh_scaled = copy.deepcopy(self.init_mesh).transform(self.Ts)
        o3d.io.write_triangle_mesh('../results/' + data_folder + "/{}_init.obj".format(self.type), self.init_mesh_scaled,  write_ascii=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)

class LOD2(LOD):
    def __init__(self, LOD_mesh) -> None:
        super().__init__(LOD_mesh)
       

class LOD3(LOD):
    def __init__(self, LOD_mesh) -> None:
        super().__init__(LOD_mesh)
    def clone_LOD2_T(self, LOD2):
        self.Ts = LOD2.Ts
        self.T_fac_n = LOD2.T_fac_n
        self.T_init = LOD2.T_init
    
class Opening:

    def __init__(self):
        self.id = -1
        self.facade = -1
        self.plane = -1
        self.coord = None
        self.d2int = [] 
        self.redundant = False
        self.view = ""

class Crack:
    def __init__(self):
        self.view = ""
        self.coord2d = None
        self.coord = None
        self.plane = None
        self.id_cr_intersect = None
        self.kinematics = None

class Plane:
    def __init__(self, mesh):
        self.id = -1
        self.mesh = mesh
        self.edges = None
        self.contour = None
        self.openings = []
        self.params = None
        self.type = ""
        self.parallel_to = -1
        self.plane_line_segments = []
        self.plane_vertices = []
        self.plane_vertices_ordered = []
        self.third_coord = None
        self.extrude_dir = None
    
    def get_edges(self):
        self.edges = trimesh2edges(self.mesh)
    
    def get_contour(self):
        edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
        self.contour = o3dtut.edges_to_lineset(self.mesh, edges, (0,1,0))
    
    def plot_mesh(self):
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([self.mesh, coord_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
    
    def plot_contour(self):
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([self.contour, coord_mesh])
    
    def decimate(self, target_number_of_triangles=5, automatic=False, threshold_area_change = 0.1): #!
        if not automatic:
            mesh_smp = self.mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
            self.mesh = mesh_smp
        else:
            initial_surface_area = self.mesh.get_surface_area()
            targ_tri = len(self.mesh.triangles)-1
            check_area_change=True
            while check_area_change:
                mesh_smp = self.mesh.simplify_quadric_decimation(target_number_of_triangles=targ_tri)
                targ_tri-=1
                current_surface_area = mesh_smp.get_surface_area()
                area_chage = 100*np.abs((initial_surface_area - current_surface_area)/initial_surface_area)
                if area_chage>threshold_area_change:
                    check_area_change=False
                else:
                    self.mesh = mesh_smp
            #print("initial area: {}; current area: {}".format(initial_surface_area, self.mesh.get_surface_area()))
    
    def get_contour_lines_ordered(self):
        #THis function will take contour of the plane, organize the lines and vertices
        plane_contour = np.array(self.contour.lines)
        plane_contour_copy = copy.deepcopy(plane_contour)
        plane_contour_ordered = []
        #ordering lines contour to make polygon with shapely
        for i, c in enumerate(plane_contour):
            if i==0:
                plane_contour_ordered.append(c)
                plane_contour_copy = np.delete(plane_contour_copy,0,0)
            else:
                id_next_line = np.where(plane_contour_copy==plane_contour_ordered[-1][1])[0]
                id_next_v = np.where(plane_contour_copy[id_next_line][0]!=plane_contour_ordered[-1][1])[0]
                plane_contour_ordered.append(np.array([plane_contour_ordered[-1][1], plane_contour_copy[id_next_line, id_next_v][0]]))
                plane_contour_copy = np.delete(plane_contour_copy, id_next_line, 0)
        plane_contour_ordered = np.array(plane_contour_ordered)
               
        #Select the intersections that are over the facade contour using vector rejection https://en.wikipedia.org/wiki/Vector_projection
        plane_vertices = np.round(np.array(self.mesh.vertices),4)
        dir_id = np.array(range(3))
        dir_id = np.delete(dir_id, self.parallel_to)
        self.third_coord = plane_vertices[0, self.parallel_to]
        plane_vertices = plane_vertices[:,dir_id]
        plane_vertices_ordered = plane_vertices[plane_contour_ordered[:,0]]
        plane_line_segments = np.concatenate((plane_vertices[plane_contour_ordered[:,0]], plane_vertices[plane_contour_ordered[:,1]]), axis=1)

        #MODIDYING CONTOUR SEQUENCE TO DELETE INTERMEDIARY POINTS. REQUIRED FOR GENERATING THE GRAPH
        #In case that the contour given by o3d has straight lines with various nodes between extrems
        #it is necessary to delete te intermediary nodes of the facade_line_segments variable
        for jj in range(2):
            lap_dir = None
            when_dir_changed = [0,]
            dirs_ = []
            for ii, fls in enumerate(plane_line_segments):
                if lap_dir is None:
                    lap_dir = fls[:2]-fls[2:]
                    lap_dir = lap_dir/np.linalg.norm(lap_dir)
                    dirs_.append(lap_dir)
                else:
                    new_lap_dir = fls[:2]-fls[2:]
                    new_lap_dir = new_lap_dir/np.linalg.norm(new_lap_dir)
                    if np.linalg.norm(new_lap_dir-lap_dir)>2e-3:
                        when_dir_changed.append(ii)
                    lap_dir = new_lap_dir
                    dirs_.append(lap_dir)
            plane_line_segments_new = np.empty(shape=(0,4))
            for ii, wdc in enumerate(when_dir_changed):           
                if wdc==0:
                    #Start with the last change. Might be a problem if the initial node is intermediary. It is required to compute twice the when_dir_change then.
                    l_seg = np.zeros((1,4))
                    l_seg[0,:2] = plane_line_segments[when_dir_changed[-1],:2]
                    l_seg[0,2:] = plane_line_segments[wdc,:2]
                else:
                    l_seg = np.zeros((1,4))
                    l_seg[0,:2] = plane_line_segments[when_dir_changed[ii-1],:2]
                    l_seg[0,2:] = plane_line_segments[wdc,:2]
                plane_line_segments_new = np.concatenate((plane_line_segments_new, l_seg), axis=0)
            plane_line_segments = plane_line_segments_new

        #Select the intersections that are over the facade contour using vector rejection https://en.wikipedia.org/wiki/Vector_projection
        self.plane_line_segments = plane_line_segments
        self.plane_vertices = plane_vertices
        self.plane_vertices_ordered = plane_line_segments[:,:2]
    
class line_fo: #line facade_openings:
    def __init__(self):
        self.plane_id = -1
        self.id = -1
        self.id_global = -1
        self.params = None #(a,b,c) @ x_h = 0
        self.type = "" #facade contour or openings contour "f", "o"
        self.dir = "" #direction of the line. vertical or horizontal for facade and openings. diagonal for facade
        self.inte_list = [] #list of intersections
        self.dir_id = None
        self.third_coord = None

class inter:
    def __init__(self):
        self.plane_id = -1
        self.id = -1
        self.id_global = -1
        self.coord = None
        self.coord_3d = None
        self.lines_id = []
        self.over_facade = False
        self.contour = False     
        self.vertex = False   
        self.redundant = False

class poly_cell:
    def __init__(self):
        self.plane_id = -1
        self.id = -1 # maybe do (i,j) line, column. To iterate over it and assign class
        self.inter_list = []
        self.type = '' #opening, pier, spandrel or node p,s,n,o
        self.t = -1
        self.b = -1
        self.r = -1
        self.l = -1
        self.tl = -1
        self.tr = -1
        self.bl = -1
        self.br = -1
        self.macro_element_id = -1
        self.split = False

class macro_element:
    def __init__(self):
        self.type = "" #p, s
        self.cells = [] #list with polycells 
        self.ij_nodes = [] #macro_nodes i and j connected by element. p: i->botoom j-> top; s: i->left j->right #!
        self.id = -1
        self.split = False
        self.b = None #dimension b (pier:horizontal; spandrel:vertical)
        self.h = None #dimension h (pier:vertical; spandrel:horizontal)
        self.c = None #point c coord (element center coords)
        self.c_local = None
        self.t = "" #neighboors elements id
        self.b = ""
        self.l = ""
        self.r = ""
        self.wall = -1
        self.contour = False
        self.contour_type = ""

class node:
    def __init__(self):
        self.cells = [] #list with polycells 
        self.type = "" #2d or 3d
        self.contour = False #if the node is a facade contour node or not
        self.contour_type = "" #top, bottom, left, right, tl, tr, bl, br
        self.t = [] #neighboors elements id
        self.b = []
        self.l = []
        self.r = []
        self.coord = None #coordinate for macroelement geometry
        self.coord_local = None #local coordinate for macroelement geometry
        self.c = None #point c coord (element center coords)
        self.id = -1
        self.redundant = False #3d nodes are connected to other 3d node. Just one of them has to be kept to generate the geo file. The other is redundant
        self.redundant_id = -1   
        self.shape = "N" #tremuri node shape. N:no shape, R:rectangular
        self.xl = 0. #R nodes in thremuri are defined using xlelf, xright, zup and zdown dimentions init as 0. in case of nodes without mass
        self.xr = 0.
        self.zu = 0.
        self.zd = 0.
        self.repart = []
        self.restricted = False
        self.wall = -1
        self.vanished = False

class opening:
    def __init__(self):
        self.cells = [] #list with polycells
        self.id = -1 

class wall:
    def __init__(self):
        self.id = -1
        self.plane_id = -1
        self.parallel_to = -1
        self.third_coord = None
        self.init = None
        self.end = None
        self.coord = None #origin of local coord
        self.angle = None
        self.nodes = []
        self.elements = []
        self.openings = []
        self.rigid_elements = []
        