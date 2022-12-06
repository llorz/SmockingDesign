function fval = energy_maximize_embedding(X, E_eq, E_neq, w_eq, w_neq, eps_neq)
if nargin < 6, eps_neq = -1e-3; end

D1 = pdist2(X, X, 'euclidean'); % the pairwise distance from the current embedding

get_mat_entry = @(M,I,J) M(sub2ind(size(M),I,J));

fval = -sum(D1(:));

d_eq = get_mat_entry(D1, E_eq(:,1), E_eq(:,2));
d_neq = get_mat_entry(D1, E_neq(:,1), E_neq(:,2));

% fval = -sum(d_neq);
% preserve edge distance
fval = fval + w_eq*sum((d_eq - E_eq(:,3)).^2);

% not exceed the distance constraint
% fval = fval + w_neq*sum((d_neq - E_neq(:,3)) > eps_neq);

end