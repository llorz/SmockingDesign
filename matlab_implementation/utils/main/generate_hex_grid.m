function [X, E, vtx_type] = generate_hex_grid(numx, numy, R, eps)
if nargin < 3 
    R = 1;
end

if nargin < 4 
    eps = 1e-6;
end



h = sqrt(3)*R;


% center of the hex
cx = 3*R*(0:numx-1);
cy = h*(0:numy-1);


% hex points
hp = [];

for ix = 1:length(cx)
    for iy = 1:length(cy)

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


% add center to the missing cell
cx_mid = 1.5*R  +  3*R*(0:numx-2);
cy_mid = -0.5*h + h*(0:numy);


[gx, gy] = meshgrid(cx, cy);
[gx_mid, gy_mid] = meshgrid(cx_mid, cy_mid);
% hex centers
hc = [[gx(:), gy(:)]; [gx_mid(:), gy_mid(:)]];


% add edges
X = [hc; hp];
% zero: center point; one: hex border point
vtx_type = [zeros(size(hc,1),1); ones(size(hp,1), 1)]; 

D = squareform(pdist(X));
ind = find(D < R+eps);
[I, J] = ind2sub(size(D), ind);

E = sort_edge([I(:), J(:)]);
end