function [] = plot_unit_pattern(usp)
base_x = usp.base_x;
base_y = usp.base_y;
sewingLines = usp.sewingLines;
sewingLinesID = 1:length(sewingLines);




mycolor = lines(100);

[X,Y] = meshgrid(0:base_x, 0:base_y);
F = zeros(size(X));



% plot the grid
surf(X,Y,F,'EdgeColor','k','FaceColor','k', 'FaceAlpha', 0.05); hold on;
% add the sewing lines
for i = 1:length(sewingLines)
    plot(sewingLines{i}(:,1), sewingLines{i}(:,2), ...
        'LineWidth', 2, 'Color', mycolor(sewingLinesID(i),:));
end
axis equal;
view([0,90])
set(gca,'XAxisLocation','top','YAxisLocation','left');
set(gca, 'YTIck', 0:base_y)
set(gca, 'XTick', 1:base_x)
title(['unit smocking pattern: ', usp.name])


end
