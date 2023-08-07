function [] = plot_embedded_smocked_graph(SG, X, ifShowPleat)

% underlay edges
for i = reshape(SG.eid_underlay,1, [])
    edge = X(SG.E(i,:), :);
    plot3(edge(:,1), edge(:,2), edge(:,3), ...
        'LineWidth',1,'Color','r'); hold on;
end
if ifShowPleat

    % pleat edges
    for i = reshape(SG.eid_pleat,1, [])
        edge = X(SG.E(i,:), :);
        plot3(edge(:,1), edge(:,2), edge(:,3), ...
            'LineWidth',1,'Color','k'); hold on;
    end
    % pleat vertices
    vid = SG.vid_pleat;
    scatter3(X(vid,1), X(vid,2), X(vid, 3),50, 'filled','k'); axis equal; hold on;

end


% underlay vertices
vid = SG.vid_underlay;
scatter3(X(vid,1), X(vid,2), X(vid, 3),100, 'filled','r'); axis equal; hold on;

axis off;
grid on;
view([0,90])

end