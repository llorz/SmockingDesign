function [F, U, UV, grid_V, vid] = arap_simulate_smocked_design(SP, SG, X, grid_step, eps_node)
if nargin < 5, eps_node = 0; end

% [gx, gy]= meshgrid(-2*grid_step:grid_step:SP.len_x+2*grid_step,...
%     -2*grid_step:grid_step:SP.len_y+2*grid_step);


[gx, gy]= meshgrid(-0.5:grid_step:SP.len_x+0.5,...
    -0.5:grid_step:SP.len_y+0.5);

[V, ~, F_quad, ~] = extract_graph_from_meshgrid(gx, gy);

grid_V = V;

UV = V;
UV = UV - min(UV);
UV = UV./max(UV(:));

V = [V, zeros(size(V,1), 1)];


F = [F_quad(:, 1:3); F_quad(:, [3,4,1])];
F = F(:,[3,2,1]);


vid = knnsearch(V(:, [1,2]), SP.V); % these nodes are fixed to X

V_new = X(SG.V_sp2sg(:,2), :);

if eps_node > 0 %
    V_new = X(SG.V_sp2sg(:,2), :);
    for sid = 1:length(SP.sewingPtsID)
        ii = SP.sewingPtsID{sid};
        for jj = 2:length(ii)
            normv = @(x) x/norm(x);
            V_new(ii(jj), :) = V_new(ii(jj), :) + [normv(SP.V(ii(jj), :) - SP.V(ii(jj-1), :))*eps_node, 0];
        end
    end
end
U = arap(V, F, vid, V_new);


end