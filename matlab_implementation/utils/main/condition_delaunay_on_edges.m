function [T_new, E_new]= condition_delaunay_on_edges(V, E)
% V: vtx positions
% E: generate delaunay triangulation conditioned on this edge list

T = delaunay(V(:,1), V(:,2));


E = sort_edge(E);

ET = [T(:, [1,2]); T(:, [2,3]); T(:, [1,3])];
ET = sort_edge(ET);


% the edges that are not part of the triangulation
eid_to_fix = find(~ismember(E, ET,  'rows'));

T_new = T;
for eid = reshape(eid_to_fix, 1, [])
    
    % find the edge in ET that is interescting with eid
    
    
    vid1 = E(eid, :);
    fids = find(sum(ismember(T, E(eid, :)), 2)); % find the faces that are adjacent to this edge
    
    intersect_vids = [];
    
    for curr_fid = reshape(fids, 1, [])
        vid2 = setdiff(T(curr_fid, :), vid1);
        
        flag = find_intersection_of_two_line_segments(V(vid1(1), :), V(vid1(2), :), V(vid2(1),: ), V(vid2(2), :));
        if flag
            intersect_vids = vid2;
            break
        end
    end
    
    % find the two faces to be updated
    update_fids = find(sum(ismember(T, intersect_vids), 2) == 2);
    
    T_new(update_fids, :) = [vid1, intersect_vids(1); vid1, intersect_vids(2)];
end



E_new= [T_new(:, [1,2]); T_new(:, [2,3]); T_new(:, [1,3])];
E_new = sort_edge(E_new);

eid_check = find(~ismember(E, E_new,  'rows'), 1);
if ~isempty(eid_check)
    error('Something is wrong...')
end
end