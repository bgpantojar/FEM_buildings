#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 20:06:06 2020

@author: pantoja
"""
import os
import sys
sys.path.append("../src")
import numpy as np
import Part
import FreeCAD
from FreeCAD import Base
import importOBJ
#from opening_builder import *
#from amuya.src.opening_builder import *
from opening_builder import *

##################################USER INTERACTION#############################
###############USER INPUT
data_folder =  'p4_09_Country_house_4'
data_path = "../results/" + data_folder 
#############USER CALLING FUNCTIONS
print("Building LOD3")
#Builder
op_builder(data_folder, data_path)
