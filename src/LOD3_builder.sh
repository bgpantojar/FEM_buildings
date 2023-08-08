#!/bin/bash
#FreeCAD path
export FREECADPATH=/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/13_AGEFM/02_repositories/FEM_buildings/freecad_dev/usr/bin

$main_LOD3_file=$1

PYTHONPATH=${LODPATH} PATH=$PATH:${FREECADPATH} freecadcmd $main_LOD3_file $@

#From terminal, being inside src folder, activate environment and run in terminal as:  
#./LOD3.sh ../examples/p2_LOD3_00_School_main_LOD3.py
