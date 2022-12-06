function [] = plot_smocking_pattern(base_x, base_y, sewingLines, sewingLinesID)
if nargin < 4
    sewingLinesID = 1:length(sewingLines);
end


mycolor = rand(length(sewingLines), 3);
mycolor = lines(100);

[X,Y] = meshgrid(0:base_x, 0:base_y);
F = zeros(size(X));


figure(1); clf;
% plot the grid
surf(X,Y,F,'EdgeColor','k','FaceColor','k', 'FaceAlpha', 0.05); hold on;
% add the sewing lines
for i = 1:length(sewingLines)
    plot(sewingLines{i}(:,1), sewingLines{i}(:,2), ...
        'LineWidth', 2, 'Color', mycolor(sewingLinesID(i),:));
end
axis equal;
view([0,90])
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
set(gca, 'YTIck', 0:base_y)
set(gca, 'XTick', 1:base_x)
end
%%
[X,Y] = meshgrid(0:10, 0:10);
 [grid_V, grid_E] = extract_graph_from_meshgrid(X, Y, true)