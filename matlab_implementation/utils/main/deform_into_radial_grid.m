function  [SP2, radius] = deform_into_radial_grid(SP, scale)
if nargin < 2
    scale = 2*pi/SP.len_x; % increase of radius to make the grid cell as square as possible
end

V = SP.V;
V_new = zeros(size(V));


theta = reshape(linspace(0, 2*pi, SP.len_x + 1),[], 1);
radius = [SP.len_x/(2*pi)];
check_y = -1:SP.len_y;
for yid = 1:length(check_y)
    vid = find(V(:, 2) == check_y(yid));
    V_new(vid, :) = [radius(yid)*cos(theta(V(vid, 1)+1)), radius(yid)*sin(theta(V(vid, 1)+1))];
    radius(end+1) = radius(end)*(1 + scale);
end

%%
vid1 = find(V(:, 1) == 0);
vid2 = find(V(:, 1) == SP.len_x);
% double check vid2 should be merged into vid1
if sum(sum((V_new(vid1, :) - V_new(vid2, :)).^2, 2)) >  1e-12
    error('wrong radial grid mapping!')
end
% remove vid2 and establish the correspondence ref
V_corres = zeros(size(V_new, 1), 2);
V_corres(:,1) = 1:size(V_new, 1);
vid_rest = setdiff(1:size(V_new, 1), [vid1(:); vid2(:)]);
V_corres(vid1, 2) = 1:length(vid1);
V_corres(vid2, 2) = 1:length(vid2);
V_corres(vid_rest, 2) = (1:length(vid_rest)) + length(vid1);

V_clean = [V_new(vid1, :); V_new(vid_rest, :)];

%%
% create grid_x and grid_y just to draw the grid faster
% they have duplicated vertices
% do not use them in any computations!
grid_x = reshape(V_new(:, 1), SP.len_y +2, SP.len_x + 1);
grid_y = reshape(V_new(:, 2), SP.len_y +2, SP.len_x + 1);
grid_z = zeros(size(grid_x));
%%
SP2 = SP;
SP2.grid_x = grid_x;
SP2.grid_y = grid_y;
SP2.grid_V = V_clean;
SP2.V = V_clean;
SP2.all_sewingPts = unique(V_corres(SP.all_sewingPts, 2));
%% update the vids in edge
E = SP.E;
[~, id] = ismember(E, V_corres(:, 1));
E_new = [V_corres(id(:,1),2), V_corres(id(:,2), 2)];
E_new = [E_new; E_new(:, [2,1])];
E_new = E_new(E_new(:,1) < E_new(:, 2),:);
E_new = unique(E_new, 'rows');
SP2.E = E_new;



E = SP.grid_E;
[~, id] = ismember(E, V_corres(:, 1));
E_new = [V_corres(id(:,1),2), V_corres(id(:,2), 2)];
E_new = [E_new; E_new(:, [2,1])];
E_new = E_new(E_new(:,1) < E_new(:, 2),:);
E_new = unique(E_new, 'rows');
SP2.grid_E = E_new;



E = SP.grid_Ediag;
[~, id] = ismember(E, V_corres(:, 1));
E_new = [V_corres(id(:,1),2), V_corres(id(:,2), 2)];
E_new = [E_new; E_new(:, [2,1])];
E_new = E_new(E_new(:,1) < E_new(:, 2),:);
E_new = unique(E_new, 'rows');
SP2.grid_Ediag = E_new;
%% update vids in faces
F = SP.grid_F;
[~, id] = ismember(F, V_corres(:, 1));
F_new  = [];
for i = 1:size(F, 2)
F_new = [F_new, V_corres(id(:, i), 2)];
end
SP2.grid_F = F_new;

%% update vids in sewingLines
SP2.sewingPtsID = cellfun(@(vids) V_corres(vids, 2), SP.sewingPtsID, 'uni',0); % should not be any degenerated case
SP2.sewingLines= cellfun(@(vids) V_clean(vids, :), SP2.sewingPtsID, 'uni', 0);
end
