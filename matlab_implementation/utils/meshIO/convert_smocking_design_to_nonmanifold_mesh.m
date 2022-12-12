function [F_new, V_new, IA, IC] = convert_smocking_design_to_nonmanifold_mesh(SD)
% SD is a solved smocking design;
SP = SD.SP;

% the stitching points in the pattern will be merged

% find the corresponding vtx in the fine grid SD.

all_vid_c = cellfun(@(vid) knnsearch(SD.grid_V, SP.grid_V(vid,:)),...
    SP.sewingPtsID, 'uni', 0);

vtx_corres = (1:size(SD.grid_V, 1))';
for sid = 1:length(all_vid_c)
    vtx_corres(all_vid_c{sid}) = all_vid_c{sid}(1);
end

[~,IA,IC] = unique(vtx_corres);

F_new = IC(SD.F);
V_new = SD.curr_V(IA,:);

end