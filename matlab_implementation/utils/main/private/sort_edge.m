function E = sort_edge(E)
func_sort_edge = @(E) unique(...
    [min(E, [], 2), max(E, [], 2)],...
    'rows');
E = func_sort_edge(E);
eid = E(:, 1) < E(:, 2);
E = E(eid, :);
end