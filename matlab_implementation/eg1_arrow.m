clc; clear; cla;
addpath(genpath('utils/'));

%% load the unit smocking pattern
usp_dir = '../unit_smocking_patterns/';
usp_name = 'arrow';

usp = read_usp([usp_dir, usp_name, '.usp']);

figure(1);  clf;
plot_unit_pattern(usp);

%% generate the full smocking pattern by tiling
% tile the unit pattern
numX = 3; 
numY = 4;
SP = SmockingPattern(usp, numX, numY);

figure(2); clf;
plot(SP);

%% our algorithm 
SD = solve_smocking_design(SP);

figure(4); clf;
trimesh(SD.F, SD.curr_V(:,1), SD.curr_V(:,2), SD.curr_V(:,3)); 
axis equal; axis off; view([0,90]);
title('simulated')


