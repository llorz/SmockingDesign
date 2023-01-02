function [X, E, vtx_type] = generate_hex_grid_circle(R, num_layer)


h = sqrt(3)*R;



% center of the hex
cx = 3*R*(-num_layer:num_layer);
cy = h*(-num_layer:num_layer);

cx_mid = 1.5*R  + cx;
cy_mid = -0.5*h + cy;

% cx = [cx(:); cx_mid(:)];
% cy = [cy(:); cy_mid(:)];
cx_keep = [];
cy_keep = [];
% hex points
hp = [];

for ix = 1:length(cx)
    for iy = 1:length(cy)

        if cx(ix)^2 + cy(iy)^2 <= (num_layer*h)^2
            cx_keep(end+1) = cx(ix);
            cy_keep(end+1) = cy(iy);

            for theta = [60,120,180,240,300,360]
                x = R * cosd(theta) + cx(ix);
                y = R * sind(theta) + cy(iy);
                if isempty(hp)
                    hp(end+1, :) = [x, y];
                else
                    [~, err] = knnsearch(hp, [x,y]);
                    if err > eps
                        hp(end+1, :) = [x, y];
                    end
                end
            end
        end
    end
end




hc = [cx_keep(:), cy_keep(:)];


% add edges
X = [hc; hp];
% zero: center point; one: hex border point
vtx_type = [zeros(size(hc,1),1); ones(size(hp,1), 1)];

D = squareform(pdist(X));
ind = find(D < R+eps);
[I, J] = ind2sub(size(D), ind);

E = sort_edge([I(:), J(:)]);
end