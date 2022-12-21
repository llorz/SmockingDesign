function SP2 = add_margin_to_smocking_pattern(SP, m_left, m_right, m_bottom, m_top)

x_ticks = unique(SP.V(:,1));
y_ticks = unique(SP.V(:,2));

xx = [-m_left + min(x_ticks);...
    x_ticks(:);
    max(x_ticks) + m_right];
yy = [-m_bottom + min(y_ticks); ...
    y_ticks(:);
    max(y_ticks) + m_top];

SP2 = SP;

[SP2.grid_x, SP2.grid_y] = meshgrid(xx, yy);


SP2 = SP2.convert_grid_to_graph(false, false);
SP2 = SP2.get_sewing_points_ID();
end