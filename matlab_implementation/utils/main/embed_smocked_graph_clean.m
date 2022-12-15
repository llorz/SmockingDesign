function [X_res] = embed_smocked_graph_clean(SG, para, Xini)
options = optimoptions('fminunc','Display',para.opti_display);


% edge constraints for the underlay graph
E_underlay = SG.E(SG.eid_underlay, :);
C_underlay_eq = zeros(size(E_underlay, 1), 3);
for eid = 1:size(E_underlay, 1)
    C_underlay_eq(eid, :) = [E_underlay(eid, :), ...
        SG.compute_max_dist(E_underlay(eid, 1), E_underlay(eid, 2))];
end

% edge constraints for the pleat graph
C_pleat_eq = [SG.E(SG.eid_pleat, :), ...
    reshape(arrayfun(@(eid) SG.compute_max_edge_length(eid), SG.eid_pleat), [], 1)];

% initialization
if nargin < 3
    X_underlay_ini = SG.V(SG.vid_underlay, :);
    X_pleat_ini = [SG.V(SG.vid_pleat, :), ...
        para.pleat_height*ones(length(SG.vid_pleat), 1)];
else

    if size(Xini,1) == SG.nv
        X = Xini;
    elseif size(Xini, 1) == SG.SP.nv
        X = Xini(cellfun(@(x) x(1), SG.vid_from_sp), :);
    else
        error('wrong size for initialization')
    end
    X_underlay_ini = X(SG.vid_underlay, 1:2);
    X_pleat_ini = X(SG.vid_pleat, :);
end

%--------------------------------------------
%-    embed the underlay
%--------------------------------------------
if para.w_u_embed > 0
    myfun_underlay = @(x) para.w_u_eq*energy_preserve_edge_length(reshape(x, [] ,2), C_underlay_eq) +...
        para.w_u_embed*energy_maximize_embedding(reshape(x, [], 2));
else
    myfun_underlay = @(x) energy_preserve_edge_length(reshape(x, [] ,2), C_underlay_eq);
end

x = fminunc(myfun_underlay,...
    X_underlay_ini(:),...
    options);



X_underlay = reshape(x, [], 2);
energy_preserve_edge_length(X_underlay, C_underlay_eq)

%--------------------------------------------
%-    embed the pleat
%--------------------------------------------
myfun_pleat = @(x) para.w_p_eq*energy_preserve_edge_length([ ...
    [X_underlay, zeros(size(X_underlay,1), 1)];...
    reshape(x, [] ,3)], ...
    C_pleat_eq,...
    size(X_underlay,1));
if para.w_p_embed > 0
    myfun_pleat = @(x)  myfun_pleat(x) + para.w_p_embed*energy_maximize_embedding([ ...
        [X_underlay, zeros(size(X_underlay,1), 1)];...
        reshape(x, [] ,3)]);
end

if para.w_p_var > 0
    myfun_pleat = @(x) myfun_pleat(x) + para.w_p_var*energy_height_variance(reshape(x, [], 3));
end

if para.w_p_height > 0
    vid_pleat_height = setdiff(SG.vid_pleat, SG.vid_pleat_border) - length(SG.vid_underlay);
    myfun_pleat = @ (x) myfun_pleat(x) + para.w_p_height*energy_positive_height(x(2*length(x)/3 + vid_pleat_height));
end



x = fminunc(myfun_pleat, X_pleat_ini(:), options);


X_pleat = reshape(x, [], 3);

energy_preserve_edge_length([[X_underlay, zeros(size(X_underlay,1), 1)]; X_pleat], C_pleat_eq)

X_res = [ [X_underlay, zeros(size(X_underlay,1), 1)]; ...
    X_pleat];
end



function [fval, grad] = energy_preserve_edge_length(X, C_eq, num_fixed)
% squared distance
% d_eq = sum((X(C_eq(:,1), :) - X(C_eq(:,2), :)).^2, 2);
% compare to the preserved edge length
% err = d_eq - C_eq(:,3).^2;

d_eq = sqrt(sum((X(C_eq(:,1), :) - X(C_eq(:,2), :)).^2, 2));
err = d_eq - C_eq(:,3);

fval = sum(err.^2);



if nargout > 1
    grad = zeros(size(X));
    for eid = 1:size(C_eq, 1)
        grad(C_eq(eid,1), :) = grad(C_eq(eid,1), :) + 4*err(eid)*(X(C_eq(eid,1), :) - X(C_eq(eid,2), :));
        grad(C_eq(eid,2), :) = grad(C_eq(eid,2), :) + 4*err(eid)*(X(C_eq(eid,2), :) - X(C_eq(eid,1), :));
    end
    if nargin > 2
        grad = grad(num_fixed+1:end, :);
    end
    grad = grad(:);
end

end

function fval = energy_maximize_embedding(X)
D1 = pdist2(X, X, 'euclidean'); % the pairwise distance from the current embedding
fval = -sum(D1(:));
end


function fval = energy_height_variance(X)
fval = var(X(:,3));
end

function fval = energy_positive_height(height)
fval = sum((height <= 0));
end

