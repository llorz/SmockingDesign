function fval = energy_pleat_embedding(X_pleat, X_underlay, E_neq, w_neq, w_var, eps_neq)
if nargin < 6, eps_neq = -1e-3; end

X = [ [X_underlay, zeros(size(X_underlay,1), 1)]; ...
    X_pleat];
    
D1 = pdist2(X, X, 'euclidean'); % the pairwise distance from the current embedding

get_mat_entry = @(M,I,J) M(sub2ind(size(M),I,J));

fval = -sum(D1(:));

fval = fval + w_var*var(X_pleat(:,3));

d_neq = get_mat_entry(D1, E_neq(:,1), E_neq(:,2));

% not exceed the distance constraint
% fval = fval + w_neq*sum((d_neq - E_neq(:,3)) > eps_neq);

fval = fval + w_neq*sum((d_neq - E_neq(:,3)).^2);

end