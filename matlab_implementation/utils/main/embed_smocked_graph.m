function [X_res] = embed_smocked_graph(SG, para)
if ~isfield(para ,'pleat_height'), para.pleat_height = 1; end
if ~isfield(para, 'w_eq'), para.w_eq = 1e6; end
if ~isfield(para, 'w_neq'), para.w_neq = 1e2; end
if ~isfield(para, 'opti_display'), para.opti_display = 'iter'; end
if ~isfield(para, 'w_neq_pleat'), para.w_neq_pleat = 1e3; end
if ~isfield(para, 'w_var'), para.w_var = 1e2; end
if ~isfield(para, 'eps_neq'), para.eps_neq = -1e-6; end

%% find the distance constraints
% for each underlay node, find the original vertercies
num =  length(SG.vid_underlay);

% the max distance between two underlay nodes - constrained by the fabric
% (assume the fabric cannot be stretched)

D = zeros(num);
for i = 1:num-1
    for j = i+1:num
        dist = SG.compute_max_dist(i ,j);
        D(i,j) = dist;
        D(j,i) = dist;
    end
end

%% remove useless constraints
%  there is no need to keep all the pairs - some of the
% constraints will never be violated, so we can remove them
% e.g., we know
% dist(xi, xj) <= d_ij, dist(xi, xk) <= d_ik, dist(xj, xk) <= d_jk
% also dist(xi, xj) + dist(xi, xk) >= dist(xj, xk)
% therefore, if we have d_ij + d_ik < d_jk
% dist(xj, xk) <= d_jk will always satisfy for any embeding of the underlay
% graph in the Euclidean space, therefore, we can ignore this constraint

C = nchoosek(1:num,3); % all combinations of (i,j,k)
useless_constraints = [];
% find the useless constraints
for cid = 1:size(C, 1)
    i = C(cid, 1); j = C(cid, 2); k = C(cid, 3);
    if D(i,j) + D(i,k) < D(k,j)
        useless_constraints(end+1, :) = [k, j];
    end
    
    if D(i,j) + D(k,j) < D(i, k)
        useless_constraints(end+1, :) = [i, k];
    end
    
    if D(i,k) + D(k,j) < D(i, j)
        useless_constraints(end+1, :) = [i, j];
    end
end
% remove the duplicated ones
useless_constraints = unique( [min(useless_constraints, [] ,2), max(useless_constraints, [] ,2)], 'rows');
% the useful constraints
dist_constraints = setdiff(nchoosek(1:num,2), useless_constraints, 'rows');

%% embed the underlay graph
% prepare the equality constraints
E_eq = SG.E(SG.eid_underlay, :);
% prepare the inequality constraints
E_neq = setdiff(dist_constraints, E_eq, 'rows');
get_mat_entry = @(M,I,J) M(sub2ind(size(M),I,J));
E_eq = [E_eq, get_mat_entry(D, E_eq(:,1), E_eq(:,2))];
E_neq = [E_neq, get_mat_entry(D, E_neq(:,1), E_neq(:,2))];
%%
X_underlay = SG.V(SG.vid_underlay, :);

X_pleat = [SG.V(SG.vid_pleat, :), para.pleat_height*ones(length(SG.vid_pleat), 1)];
%% find the embedding of the underlay

options = optimoptions('fminunc','Display',para.opti_display);
myfun = @(x) energy_maximize_embedding(reshape(x, [] ,2), E_eq, E_neq, para.w_eq, para.w_neq);
x0 = X_underlay(:);

x = fminunc(myfun,x0,options);

X_underlay = reshape(x, [], 2);
%%
E_pleat_neq = [SG.E(SG.eid_pleat, :), ...
    reshape(arrayfun(@(eid) SG.compute_max_edge_length(eid), SG.eid_pleat), [], 1)];
%%


options = optimoptions('fminunc','Display',para.opti_display);

myfun = @(x) energy_pleat_embedding(reshape(x, [] ,3), X_underlay, E_pleat_neq, para.w_neq_pleat, para.w_var, para.eps_neq);
x0 = X_pleat(:);

x = fminunc(myfun,x0,options);

X1 = reshape(x, [], 3);
%%
X_res = [ [X_underlay, zeros(size(X_underlay,1), 1)]; ...
    X1];
end