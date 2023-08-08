%% options
%%%% change this to the folder where you have Opensees.exe and Macroelement3d.ddl
analysisFolder = '/home/pantojas/Bryan_PR2/04_TAPEDA_BP/04_UAV2EFM/13_AGEFM/02_repositories/FEM_buildings/src';

% file containing all information of the model (must contain the model in a variable
% with the same name)
projectName = 'EFM';  

% file containing all information of the model (must contain the model in a variable
% with the same name)
modelFile = 'modelOpensees.mat';

%% load files
addpath('utils');
load(modelFile);
eval(['model = ',modelFile(1:end-4),';']);
load('inputFiles.mat');

%% do analyses
% copy batch file in the analysis folder
currentDir = pwd;
copyfile(inputFiles(end).filename, analysisFolder);
[~,nameBatch,extBatch] = fileparts(inputFiles(end).filename);

cd(analysisFolder);
tic;
commandLine = ['!./OpenSees ', nameBatch, extBatch];
eval(commandLine);
elapsedTime = toc;
cd(currentDir);

fprintf('Total analysis time: %.2f s\n', elapsedTime);

%% read modal analyses    

if ~isempty(inputFiles(2).filename)
    
        nModes = length(inputFiles(2).outputFiles)-1;
        for kMode = 1:nModes
            result = readAnalysis(model, inputFiles(2).outputFiles(kMode).filename, 'Modal', 1);
            analysis(2).time(kMode) = kMode;
            for kNode = 1:length(result.node)
                if ~isempty(result.node(kNode).u)
                analysis(2).node(kNode).u(kMode,1) = result.node(kNode).u(1);
                analysis(2).node(kNode).v(kMode,1) = result.node(kNode).v(1);
                analysis(2).node(kNode).w(kMode,1) = result.node(kNode).w(1);
                analysis(2).node(kNode).rotx(kMode,1) = result.node(kNode).rotx(1);
                analysis(2).node(kNode).roty(kMode,1) = result.node(kNode).roty(1);
                analysis(2).node(kNode).rotz(kMode,1) = result.node(kNode).rotz(1);
                end
            end 
        end
        % read periods            
        formatString = '%f%f%[^\n\r]';

        filename = inputFiles(2).outputFiles(end).filename;
        fileID = fopen(filename,'r');

        dataArray = textscan(fileID, formatString, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'TextType', 'string',  'ReturnOnError', false);
        fclose(fileID);

        analysis(2).periods     = dataArray{:, 1};
        analysis(2).frequencies = dataArray{:, 2};
    
end  

%% plot some analyses

%analysis 3, modal
DriftType='D'
DriftLimit=1
plotAnalysis(model, analysis(2),DriftType,DriftLimit)
title ("Deformed shape");

%% save results

fname="results.mat"
save(fname)