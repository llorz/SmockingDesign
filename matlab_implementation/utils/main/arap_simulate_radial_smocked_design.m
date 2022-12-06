function [F, U, UV] = arap_simulate_radial_smocked_design(SP2, SG, X, radius, grid_step, fixed_circle_ratio, arap_numIter, eps_node)
if nargin < 5, grid_step = 0.3; end
if nargin < 6, fixed_circle_ratio = 0.3; end
if nargin < 7, arap_numIter = 2e2; end
if nargin < 8, eps_node = 0.1; end

[gx, gy]= meshgrid(min(SP2.V(:,1))-grid_step*3:grid_step:max(SP2.V(:,1)) + grid_step*3,...
    min(SP2.V(:,2))-grid_step*3:grid_step:max(SP2.V(:,2))+grid_step*3);

[V, ~, F_quad, ~] = extract_graph_from_meshgrid(gx, gy);


UV = V;
UV = UV - min(UV);
UV = UV./max(UV(:));

V = [V, zeros(size(V,1), 1)];
F = [F_quad(:, 1:3); F_quad(:, [3,4,1])];
F = F(:,[3,2,1]);

vid = knnsearch(V(:, [1,2]), SP2.V); % these nodes are fixed to X
%% find the middle region
r = sqrt(sum(V.^2, 2));
vid2 = find(r < radius(1)*fixed_circle_ratio); % their z-coord should stay zero
Aeq = sparse(1:length(vid2), 2*size(V,1) + vid2, ones(1, length(vid2)), length(vid2), 3*size(V,1));
Beq = sparse(zeros(length(vid2), 1) + eps);
Aeq = [Aeq; sparse(1:length(vid2), size(V,1) + vid2, ones(1, length(vid2)), length(vid2), 3*size(V,1))];
Beq = [Beq; sparse(zeros(length(vid2), 1) + eps)];
Aeq = [Aeq; sparse(1:length(vid2), vid2, ones(1, length(vid2)), length(vid2), 3*size(V,1))];
Beq = [Beq; sparse(zeros(length(vid2), 1) + eps)];
%%
if length(unique(vid)) ~= length(vid)
    error('chose a finer grid!')
end

V_new = X(SG.V_sp2sg(:,2), :);
%%
if eps_node > 0 %
    for sid = 1:length(SP2.sewingPtsID)
        ii = SP2.sewingPtsID{sid};
        for jj = 2:length(ii)
            normv = @(x) x/norm(x);
            V_new(ii(jj), :) = V_new(ii(jj), :) + [normv(SP2.V(ii(jj), :) - SP2.V(ii(jj-1), :))*eps_node, 0];
        end
    end
end

tic
U = arap(V, F, vid, V_new, 'Aeq', Aeq, 'Beq', Beq, 'MaxIter', arap_numIter);
toc

end

