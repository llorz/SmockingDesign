function  [SD_L] = tile_smocking_design(SD, numx, numy, para)
grid_step = para.grid_step;
% find the middle pattern from the solved design
ix = median(1:SD.SP.num_x);
iy = median(1:SD.SP.num_y);

base_x = SD.SP.base_x;
base_y = SD.SP.base_y;
ori = [(ix - 1)*base_x, (iy-1)*base_y];

% grid vtx positions of the unit smocking pattern in the middle
[grid_x, grid_y] = meshgrid(ori(1)+ (0:base_x), ori(2) + (0:base_y));
grid_V = [grid_x(:), grid_y(:)];
vids_center = knnsearch(SD.grid_V, grid_V);
vtx_graph = SD.curr_V(vids_center, :); % copy the vtx positions for the smocked graph

% finer grid vtx for the smocking design
[f_grid_x, f_grid_y] = meshgrid(ori(1)+ (0:grid_step:base_x), ori(2) + (0:grid_step:base_y));
grid_Vf = [f_grid_x(:), f_grid_y(:)];
vids_f = knnsearch(SD.grid_V, grid_Vf);
vtx_cloth = SD.curr_V(vids_f, :);

% [c, R] = pca_rigid_alignment(vtx_graph(:, [1,2]));
% vtx_graph(:, [1,2]) = (vtx_graph(:, [1,2]) - c)*R + c;
% vtx_cloth(:, [1,2]) = (vtx_cloth(:, [1,2]) - c)*R+ c;

%% create a larger smocking pattern
SP_L = SmockingPattern(SD.SP.patternName, numx, numy);
SG_L = SmockedGraph(SP_L);

nv = size(SP_L.V, 1);
Xsp = zeros(nv, 3);
Xsp_ifset = zeros(nv, 1);


% create finer grid for cloth
[Gx, Gy] = meshgrid(0:grid_step:SP_L.len_x,...
    0:grid_step:SP_L.len_y);

[grid_V, ~, F_L, ~] = extract_graph_from_meshgrid(Gx, Gy);

UV = grid_V;
UV = UV - min(UV);
UV = UV./max(UV(:));

nv = size(grid_V, 1);
curr_V = zeros(nv, 3);
curr_V_ifset = zeros(nv, 1);

%% tile the solved smocking design to a larger patch
% TODO: too slow... can be optimized a bit :( to lazy to fix it now
for ix = 1:numx
    for iy = 1:numy
        % the origin of current unit pattern
        ori = [(ix - 1)*base_x, (iy-1)*base_y];
        % tile the optimized smocked graph
        % grid vtx positions of the current unit
        [grid_x, grid_y] = meshgrid(ori(1)+ (0:base_x), ori(2) + (0:base_y));
        V = [grid_x(:), grid_y(:)];
        vids = knnsearch(SP_L.V, V);
        
        vid_fixed = find(Xsp_ifset(vids),1);
        if isempty(vid_fixed) % nothing is fixed
            t = vtx_graph(1, :); % we move the first vertex to origin
        else
            t = vtx_graph(1, :) - Xsp(vids(vid_fixed), :);
        end
        % tile the solved position to the smocking pattern
        vids_update = find(Xsp_ifset(vids) == 0);
        Xsp(vids(vids_update), :) = vtx_graph(vids_update, :) - t;
        Xsp_ifset(vids(vids_update)) = 1;
        
        % tile the araped smocking design to the fine grid
        [f_grid_x, f_grid_y] = meshgrid(ori(1)+ (0:grid_step:base_x), ori(2) + (0:grid_step:base_y));
        Vf = [f_grid_x(:), f_grid_y(:)];
        vids_f = knnsearch(grid_V, Vf);
        vids_f_update = find(curr_V_ifset(vids_f) == 0);
        curr_V(vids_f(vids_f_update), :) = vtx_cloth(vids_f_update, :) - t;
        curr_V_ifset(vids_f(vids_f_update)) = 1;
    end
end

% curr_V = pca_rigid_alignment(curr_V);

SD_L.SP = SP_L;
SD_L.SG = SG_L;
SD_L.F = F_L;
SD_L.curr_V = curr_V;
SD_L.UV = UV;
SD_L.grid_V = grid_V;
SD_L.Xsp = Xsp;

end

