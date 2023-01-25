function [SD] = solve_smocking_design(SP, para)
if nargin < 2, para = get_para_for_smocking(); end

% smocking design
SD = struct();

fprintf('solve the smocked graph...');
a = tic;
SG = SmockedGraph(SP);
t = toc(a);
fprintf('done %.6f s\n', t);

fprintf('embed the smocked graph...');
a = tic;
X = embed_smocked_graph_clean(SG, para);
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

function para = get_para_for_smocking()
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
end