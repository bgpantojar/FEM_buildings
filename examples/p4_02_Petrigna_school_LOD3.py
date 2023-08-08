#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:06:06 2020

@author: pantoja
"""
import sys
sys.path.append("../src")
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ
from opening_builder import *

##################################USER INTERACTION#############################
###############USER INPUT
data_folder =  'p4_02_Petrigna_school'
data_path = "../results/" + data_folder 
#############USER CALLING FUNCTIONS
print("Building LOD3")
op_builder(data_folder, data_path)

