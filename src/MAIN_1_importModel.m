% creates the 3d model of the building starting from the Tremuri model
addpath('utils');

modelTremuri = readTremuriInput(tremuri_txt);
modelOpensees = convertTremuriToOpenSees(modelTremuri);
figure; hold on;
drawModelTremuri(modelTremuri, 'ColorPiers', [0.2 0.2 1.0], 'ColorSpandrels', [0.2 1. .2], 'styleNodes', 'sk')
axis equal
axis off
view( -30,   40);

figure; hold on;
drawModel(modelOpensees, 'ColorPiers', [0.2 0.2 1.0], 'ColorSpandrels', [0.2 1 0.2], 'styleNodes', 'sk', 'floors', true, 'ColorBeams', [1 1 1]*0.35, 'sizeBeams', 0.35);
axis equal
axis off
view( -30,   40);

save('modelTremuri.mat', 'modelTremuri');
save('modelOpensees.mat', 'modelOpensees');
