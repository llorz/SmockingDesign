function [grid_V, grid_E, grid_F, grid_Ediag] = extract_graph_from_meshgrid(X, Y, ifValidate)
if ~isequal(size(X), size(Y))
    error('Invalid input - not a grid!')
end

if nargin < 3, ifValidate = false; end

num = size(X,1)*size(X,2);
ind = reshape(1:num, size(X, 1), size(X,2));

grid_E = [];
grid_Ediag = [];
grid_F = [];

for i = 2:size(X, 1)
    for j = 2:size(X, 2)
        grid_E(end+1,:) = [ind(i-1,j), ind(i,j)];
        grid_E(end+1,:) = [ind(i,j-1), ind(i,j)];
        grid_Ediag(end+1,:) = [ind(i-1, j-1), ind(i,j)];
        grid_Ediag(end+1,:) = [ind(i-1, j), ind(i,j-1)];
        grid_F(end+1,:) = [ind(i-1, j-1), ind(i, j-1), ind(i,j), ind(i-1, j)];
    end
end

for j = 2:size(X, 2)
    i = 1;
    grid_E(end+1,:) = [ind(i,j-1), ind(i,j)];
end

for i = 2:size(X, 1)
    j = 1;
    grid_E(end+1,:) = [ind(i-1,j), ind(i,j)];
end

grid_V = [X(:), Y(:)];
%% visualize the extracted graph
if ifValidate
    figure(100); clf;
    
    % plot the input meshgrid
    subplot(1,2,1);
    surf(X,Y, zeros(size(X)), ...
        'EdgeColor','k','FaceColor','k', 'FaceAlpha', 0.05);
    axis equal;
    view([0,90])
    set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
    xlim([min(X(:)), max(X(:))]);
    ylim([min(Y(:)), max(Y(:))]);
    title('input grid');
    
    subplot(1,2,2);
    scatter(grid_V(:,1), grid_V(:,2),'filled'); hold on;
    
    for i = 1:size(grid_E,1)
        plot(grid_V(grid_E(i,:),1),  grid_V(grid_E(i,:),2));
    end
  
    axis equal;
    view([0,90])
    set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
    xlim([min(grid_V(:,1)),max(grid_V(:,1))]);
    ylim([min(grid_V(:,2)),max(grid_V(:,2))]);
      title('extracted graph');
end
end