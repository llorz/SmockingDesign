function [SP_L, SG_L, Xsp_L] = get_initial_embedding_by_tiling(SP, SG, X, numx, numy)
Xsp = X(SG.V_sp2sg(:,2), :);

SP_L = SmockingPattern(SP.usp, numx*SP.num_x, numy*SP.num_y);
SG_L = SmockedGraph(SP_L);
nv = size(SP_L.V, 1);
Xsp_L = zeros(nv, 3);
Xsp_L_ifset = zeros(nv, 1);

base_x = SP.len_x; 
base_y = SP.len_y;



for ix = 1:numx
    for iy = 1:numy
        % the origin of current unit pattern
        ori = [(ix - 1)*base_x, (iy-1)*base_y];
        % tile the optimized smocked graph
        % grid vtx positions of the current unit
        [grid_x, grid_y] = meshgrid(ori(1)+ (0:base_x), ori(2) + (0:base_y));
        V = [grid_x(:), grid_y(:)];
        vids = knnsearch(SP_L.V, V);
        
        vid_fixed = find(Xsp_L_ifset(vids),1);
        if isempty(vid_fixed                ) % nothing is fixed
            t = Xsp(1, :); % we move the first vertex to origin
        else
            t = Xsp(1, :) - Xsp_L(vids(vid_fixed), :);
        end
        % tile the solved position to the smocking pattern
        vids_update = find(Xsp_L_ifset(vids) == 0);
        Xsp_L(vids(vids_update), :) = Xsp(vids_update, :) - t;
        Xsp_L_ifset(vids(vids_update)) = 1;
    end
end
end