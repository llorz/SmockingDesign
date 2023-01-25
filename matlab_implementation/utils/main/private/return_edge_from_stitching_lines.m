function [E, V] = return_edge_from_stitching_lines(SL, V)
% Input:
%           V:  list of vertices
%           SL: stitching lines, where SL{i} stores a list of vtx pos
% Output:
%           E: the edge list with index from V that corresponds to SL

if nargin < 2
    V = cell2mat(SL'); % collect vtx from the stitcing lines
end

Xid = cellfun(@(sewingLine) return_nth_output(2, ....
    @ismember, ...
    sewingLine, V,'rows'),...
    SL, 'UniformOutput', false);
E2 = cell2mat(Xid)';
E = [];
for i = 1:size(E2, 2)-1 % each stitching line might contain more than 2 vertices
    E = [E; E2(:, i:i+1)];
end

E = [min(E,[], 2), max(E, [], 2)];
E = unique(E, 'rows');
end