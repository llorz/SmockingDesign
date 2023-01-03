clc; clear;
addpath(genpath('../utils/'))

usp = get_unit_smocking_pattern('braid');
SP = SmockingPattern(usp, 3, 3, 'regular',false, [0,0], false);
SG  = SmockedGraph(SP);

figure(1); clf;
plot(SP,1)


figure(2); clf;
plot(SG, 1);
% vid = SG.vid_pleat_border; X = SG.V;
% scatter(X(vid,1 ), X(vid,2),'filled'); % visualize the border
%%
para = struct();
para.pleat_height = 1;
para.w_u_eq = 1;
para.w_u_embed = 0;
para.w_p_eq = 1e3;
para.w_p_embed = 1;
para.w_p_var = 1;
para.w_p_height = 0;
para.opti_display = 'off';

tic
[X, C_underlay_eq, C_pleat_eq, X_underlay_ini, X_pleat_ini, X_underlay, X_pleat] = embed_smocked_graph_export(SG, para);
toc
%%
figure(6); clf;
scatter3(X(:,1), X(:,2), X(:, 3), 'filled'); axis equal; hold on;

for i = 1:SG.ne
    edge = X(SG.E(i,:), :);
    plot3(edge(:,1), edge(:,2), edge(:,3)); hold on;
end

axis off;
grid on;
title('Embedded Smocked Graph')
view([0,90])

vid_pleat_height = setdiff(SG.vid_pleat, SG.vid_pleat_border);
min(X(vid_pleat_height, 3));
%%
figure(4)
for i = reshape(SG.eid_underlay, 1, [])
    edge = X(SG.E(i,:), :);
    plot3(edge(:,1), edge(:,2), edge(:,3),'red','LineWidth',1); hold on;
end

for i = reshape(SG.eid_pleat, 1, [])
    edge = X(SG.E(i,:), :);
    plot3(edge(:,1), edge(:,2), edge(:,3),'blue','LineWidth',1); hold on;
end

%%

save_dir = ['./results/', SP.patternName, '/'];
if ~isdir(save_dir)
    mkdir(save_dir);
end

writematrix(C_underlay_eq, [save_dir, 'C_underlay_eq.txt']);
writematrix(C_pleat_eq, [save_dir, 'C_pleat_eq.txt']);
writematrix(X_underlay_ini, [save_dir, 'X_underlay_ini.txt']);
writematrix(X_underlay, [save_dir, 'X_underlay.txt']);
writematrix(X_pleat_ini, [save_dir, 'X_pleat_ini.txt']);
writematrix(X_pleat, [save_dir, 'X_pleat.txt']);
