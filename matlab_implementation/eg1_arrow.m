clc; clear; cla;
addpath(genpath('utils/'));

%% load the unit smocking pattern
usp_dir = '../unit_smocking_patterns/';
usp_name = 'arrow';

usp = read_usp([usp_dir, usp_name, '.usp']);

figure(1);  clf;
plot_unit_pattern(usp);

%% generate the full smocking pattern by tiling
numX = 3; 
numY = 4;
SP = SmockingPattern(usp, numX, numY);

figure(2); clf;
plot(SP);
%% our algorithm - step 01: extract the smocked graph
SG  = SmockedGraph(SP);

figure(3); clf;
plot(SG);

