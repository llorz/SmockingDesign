clc; clear; cla;
addpath(genpath('utils/'));

%% prameters for optimization
para = struct();
para.pleat_height = 1;
para.w_u_eq = 1;
para.w_u_embed = 0;
para.w_p_eq = 1;
para.w_p_embed = 1e-3;
para.w_p_var = 1e-3;
para.opti_display = 'off';
para.eps_node = 0.1;
para.grid_step = 0.1;

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

%% construct the smocked graph
SG = SmockedGraph(SP);

figure(3); clf
plot(SG); % red: underlay graph; black: pleat
%% embed the smocked graph
X = embed_smocked_graph_clean(SG, para);

figure(4); clf;
ifShowPleat = true;
plot_embedded_smocked_graph(SG, X, ifShowPleat)

