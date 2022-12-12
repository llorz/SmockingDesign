function SP_new = deform_smocking_pattern(SP, scale, t)

func_deform = @(x) scale*x + t;
SP_new = SP;
SP_new.grid_x = SP.grid_x*scale + t(1);
SP_new.grid_y = SP.grid_y*scale + t(2);
SP_new.grid_V = func_deform(SP.grid_V);
SP_new.V = func_deform(SP.V);
SP_new.sewingLines = cellfun(@(x)func_deform(x), SP.sewingLines, 'uni', 0);
end