clc
close all
clear all

%%%% change this to the folder where you have Opensees.exe and Macroelement3d.ddl
analysisFolder = '/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/13_AGEFM/02_repositories/FEM_buildings/src';

% the project name gives the name to the tcl files
projectName = 'EFM';  

% file containing all information of the model (must contain the model in a variable
% with the same name)
modelFile = 'modelOpensees.mat';

%% load files
addpath('utils');
load(modelFile);
eval(['model = ',modelFile(1:end-4),';']);

%% make directories for input and output
currentDir = pwd;
if ~exist('inputFiles', 'dir')
    mkdir('inputFiles');
end
if ~exist('outputFiles', 'dir')
    mkdir('outputFiles');
end
cd('outputFiles');
if ~exist('Modal', 'dir')
    mkdir('Modal');
end
cd('..');

%% write input files and keep track of them
inputFiles = writeOpenseesTcl(model, 'ignoreDrift', 1, 'projectName', projectName, 'massDistribution', 'Lumped', ...
    'inputPath', [currentDir, '\inputFiles'], 'outputPath', [currentDir, '\outputFiles'], 'wallToWallConnection', '1 2 3 4 5 6', 'floorToWallConnection', '1 2 3 4 5 6');

save('inputFiles.mat', 'inputFiles');