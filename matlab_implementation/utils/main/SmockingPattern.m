classdef SmockingPattern
    
    properties
    end
    
    properties
        % unit smocking pattern info
        patternName
        base_x % length of the unit pattern in x direction
        base_y % legnth of the unit pattern in y direction
        
        % constructed cloth info
        % we have a cloth of size [0, len_x] X [0, len_y]
        len_x  % length of the cloth in x direction
        len_y  % lengh of the cloth in y direction
        % grid in meshgrid format
        grid_x  % x-coord of all the grid points
        grid_y  % y-coord of all the grid points
        % grid in a graph
        grid_V % Nv-by-2 matrix, the xy-coord of the N grid points
        grid_E % Ne-by-2 matrix, the edge connectivity of the grid
        grid_Ediag % the diagonal edges in the grid
        grid_F
        V         % the same as grid_V
        E         % the combination of grid_E and grid_Ediag
        sewingLines  % a list of vertices that will be stitched together (the xy-coord of these vertices)
        sewingLines_plotID
        sewingPtsID   % the list of of vtxID as in grid_V, that will be stitched together
        edge_cid  % edge category id
        all_sewingPts
    end
    
    methods
        function S = SmockingPattern(smocking_type, num_x, num_y, mode)
            if nargin < 4, mode = 'regular'; end % regular
            % load the unit smocking pattern
            [base_x, base_y,  unit_sewingLines] = get_unit_smocking_pattern(smocking_type);
            
            % we will duplicate the unit pattern num_x times along x-axis
            % and num_y times along y-axis
            
            len_x = base_x * num_x;
            len_y = base_y * num_y;
            
            switch lower(mode)
                case 'regular'
                    [S.grid_x, S.grid_y] = meshgrid(0:len_x, 0:len_y); % add empty cells on the top
                case 'radial'
                    [S.grid_x, S.grid_y] = meshgrid(0:len_x, -1:len_y); % add empty cells on the top
            end
            % duplicate the sewingLines
            sewingLines = [];
            for  ix = 1:num_x
                for iy = 1:num_y
                    % shift the origin of the unit_sewingLines
                    sewingLines = [sewingLines, ...
                        cellfun(@(lines) lines + [(ix - 1)* base_x, (iy-1)*base_y], ...
                        unit_sewingLines, 'UniformOutput', false)];
                end
            end
            
            S.sewingLines_plotID = reshape(repmat(1:num_x*num_y, length(unit_sewingLines),1), [] ,1);
            S.sewingLines = sewingLines;
            S.len_x = len_x;
            S.len_y = len_y;
            S.patternName = smocking_type;
            S.base_x = base_x;
            S.base_y = base_y;
            
            S = S.convert_grid_to_graph();
            S = S.get_sewing_points_ID();
            S = S.categorize_grid_edges();
        end
        
        function obj = convert_grid_to_graph(obj)
            [obj.grid_V, obj.grid_E, obj.grid_F, obj.grid_Ediag] = extract_graph_from_meshgrid(obj.grid_x, obj.grid_y, false);
            obj.V = obj.grid_V;
            obj.E = [obj.grid_E; obj.grid_Ediag];
            obj.E = [min(obj.E, [], 2), max(obj.E, [], 2)];
        end
        
        function obj = get_sewing_points_ID(obj)
            % find the vtxID as in grid_V of the sewing lines
            obj.sewingPtsID =cellfun(@(sewingLine) return_nth_output(2, ....
                @ismember, ...
                sewingLine, obj.grid_V,'rows'),...
                obj.sewingLines, 'UniformOutput', false);
        end
        
        function obj = categorize_grid_edges(obj)
            vid =  cell2mat(reshape(obj.sewingPtsID, [], 1));
            obj.all_sewingPts = vid;
            % cate-1: edge not connected to any sewing points
            % cate-2: only one endpoint of the edge is a sewing points
            % cate-3: two endpoints of the edge are sewing points, but
            % belong to the same unit (i.e., these edges will be folded)
            % cate-4: two endpoints of the edge are sewing points belonging
            % to different units (i.e. the supporting edges which are rigid!)
            tmp = sum(ismember(obj.grid_E, vid),2);
            obj.edge_cid = 3*ones(size(obj.grid_E, 1), 1);
            obj.edge_cid(tmp == 0) = 1; % cate-1
            obj.edge_cid(tmp == 1) = 2; % cate-2
            eids = find(tmp == 2); % we need seperate cate-3 and cate-4
            tmp = arrayfun(@(eid) ...
                isempty(find(cellfun(@(sewingLine) sum(ismember(obj.grid_E(eid,:), sewingLine)), obj.sewingPtsID) == 2, 1)),...
                eids);  % check if two endpoints belong to the same unit
            obj.edge_cid(eids(tmp)) = 4;
        end
        
        function[] = plot(obj, mode)
            if nargin < 2, mode = 1; end
            
            mycolor = lines(100);
            mylinecolor = [0.8,0.8,0.8;...
                0.6, 0.6, 0.6; 0.4, 0.4, 0.4; 1,0,0];
            grid_z = zeros(size(obj.grid_x));
            surf(obj.grid_x, obj.grid_y, grid_z,...
                'EdgeColor','k','FaceColor','k', 'FaceAlpha', 0.05); hold on;
            if mode  == 1
                % add the sewing lines
                for i = 1:length(obj.sewingLines)
                    plot(obj.sewingLines{i}(:,1), obj.sewingLines{i}(:,2), ...
                        'LineWidth', 2, 'Color', mycolor(obj.sewingLines_plotID(i),:));
                end
                title(['Complete Smocking Pattern - ', obj.patternName])
            elseif mode == 2
                % add the sewing points
                for i = 1:length(obj.sewingPtsID)
                    scatter(obj.grid_V(obj.sewingPtsID{i}, 1), obj.grid_V(obj.sewingPtsID{i}, 2),'filled');
                end
                title('Sewing Points')
            elseif mode == 3
                % plot the grid edges w.r.t. their category
                for i = 1:size(obj.grid_E,1)
                    if obj.edge_cid(i) == 4
                        linewidth = 2;
                    else
                        linewidth = 1;
                    end
                    plot(obj.grid_V(obj.grid_E(i,:), 1), obj.grid_V(obj.grid_E(i,:), 2), ...
                        'LineWidth', linewidth, 'Color', mylinecolor(obj.edge_cid(i),:));
                end
                scatter(obj.grid_V(obj.all_sewingPts, 1), obj.grid_V(obj.all_sewingPts, 2), ...
                    10,'filled');
                title('Grid Edge Categories')
            end
            axis equal;
            view([0,90])
            set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
            set(gca, 'YTIck', 0:2:obj.len_y)
            set(gca, 'XTick', 2:2:obj.len_x)
        end
    end
    
end

