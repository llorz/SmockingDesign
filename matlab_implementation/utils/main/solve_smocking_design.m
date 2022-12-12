function [SD] = solve_smocking_design(patternName, numx, numy, para)
% smocking design
SD = struct();

fprintf('construct the smocking pattern (%s: %d-by-%d)... ', patternName, numx, numy);
a = tic; 
SP = SmockingPattern(patternName, numx, numy);
t = toc(a);
fprintf('done %.6f s\n', t);

fprintf('solve the smocked graph...');
a = tic;
SG = SmockedGraph(SP);
t = toc(a);
fprintf('done %.6f s\n', t);

fprintf('embed the smocked graph...');
a = tic;
X = embed_smocked_graph(SG, para);
t = toc(a);
fprintf('done %.6f s\n', t);


fprintf('solve the smocking design via ARAP...');
a = tic;
[SD.F, SD.curr_V, SD.UV, SD.grid_V] = arap_simulate_smocked_design(SP, SG, X, para.grid_step, para.eps_node);
t = toc(a);
fprintf('done %.6f s\n',t);

SD.SP = SP;
SD.SG = SG;
SD.Xsg = X; % the embeded positions for the smocked graph
SD.Xsp = X(SG.V_sp2sg(:,2), :); % the embedded positions frot he smocking pattern
end