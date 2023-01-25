function [F, U, UV, grid_V, vid, V_new, F_quad] = arap_simulate_smocked_design(SP, SG, X, ...
    grid_step, ...
    eps_node, ...
    margin,...
    ifsetboundryfree)

if nargin < 5, eps_node = 0; end

if nargin < 6
    margin = [0.5, 0.5, 0.5, 0.5];
end

if nargin < 7, ifsetboundryfree = false; end



[gx, gy]= meshgrid(-margin(1)+min(SP.grid_V(:,1)):grid_step:max(SP.grid_V(:,1))+margin(2),...
    -margin(3)+min(SP.grid_V(:,2)):grid_step:max(SP.grid_V(:,2))+margin(4));

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

if ifsetboundryfree
    vid_ori = cell2mat(SG.vid_from_sp(SG.vid_pleat));
    vid_boundary = vid_ori([find(SP.V(vid_ori, 1) == min(SP.V(:,1))); ...
        find(SP.V(vid_ori, 1) == max(SP.V(:, 1)));...
        find(SP.V(vid_ori, 2) == min(SP.V(:,2)));...
        find(SP.V(vid_ori, 2) == max(SP.V(:,2)));]);
    vid(vid_boundary) = [];
    V_new(vid_boundary, :) = [];
end


U = arap(V, F, vid, V_new);

% make the fixed vtx smoother...
ind2 = setdiff(1:size(U, 1), vid);
U = arap(V, F, ind2, U(ind2, :));


end