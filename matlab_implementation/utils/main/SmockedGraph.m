classdef SmockedGraph
    properties
        SP             % smocking pattern
        V                % the vertex list of the smocked graph, i.e., multiple grid vertices are merged together w.r.t. smocking pattern
        E                 % the edge list of the smocked graph
        V_sp2sg     % the corres between the grid vertices from SP and the graph vertices from SG
        vid_from_sp % vid_from_sp{i} gives the vtx from the SP that are sewed together to the i-th vtx in SG
        E_sg2sp     % the corres between the edges from SG to SP
        nv                 % number of vertices
        ne                 % number of edges
        vid_underlay % vtxID belong the underlay graph
        eid_underlay %edgeID belong to the underlay graph
        vid_pleat       % vtxID of the pleat vertices (i.e., not touching the surface)
        eid_pleat       % edgeID that forms the pleats
    end
    
    methods
        function SG = SmockedGraph(SP)
            % reindex the points
            % first column: vid in the origial grid
            % second column: new id after merging the vts
            V_corres = zeros(size(SP.grid_V));
            V_corres(:,1) = 1:size(SP.grid_V,1);
            
            vid = setdiff(1:size(SP.grid_V, 1), SP.all_sewingPts);
            num = length(SP.sewingPtsID) + length(vid);
            V_new = zeros(num, 2);
            % merge the grid vertices that belong to the same sewing line
            % set the new vertex position to the average among the gathered
            % grid vertices
            for gid = 1:length(SP.sewingPtsID)
                V_corres(SP.sewingPtsID{gid}, 2) = gid;
                V_new(gid, :) = mean(SP.grid_V(SP.sewingPtsID{gid},:));
            end
            
            V_corres(vid, 2) = (1:length(vid)) + length(SP.sewingPtsID);
            V_new(length(SP.sewingPtsID)+1:end, :) = SP.grid_V(vid, :);
            
            % update the edge
            E = SP.E; % both the grid edge and the grid diagonal edge
            [~, id] = ismember(E, V_corres(:, 1));
            E_new = [V_corres(id(:,1),2), V_corres(id(:,2), 2)];
            
            E_new = [E_new; E_new(:, [2,1])];
            E_new = E_new(E_new(:,1) < E_new(:, 2),:);
            E_new = unique(E_new, 'rows');
            
            SG.V = V_new;
            SG.E = E_new;
            SG.V_sp2sg = V_corres;
            SG.nv = size(V_new, 1);
            SG.ne = size(E_new, 1);
            
            SG.vid_underlay = 1:length(SP.sewingPtsID); % since we re-indexed the sewing points
            SG.vid_pleat = setdiff(1:SG.nv, SG.vid_underlay);
            
            SG.eid_underlay = reshape( find(sum(E_new <= length(SP.sewingPtsID), 2) == 2), 1, []);
            SG.eid_pleat = setdiff(1:SG.ne, SG.eid_underlay);
            SG.SP = SP;
            
            SG.vid_from_sp = arrayfun(@(vid) find(SG.V_sp2sg(:,2) == vid), 1:SG.nv, 'uni', 0);
            % update the endpoints index of the edges in SP w.r.t. the
            % vertex ID in SG
            E_tmp = [V_corres(SP.E(:,1),2), V_corres(SP.E(:,2), 2)];
            E_tmp = [min(E_tmp, [], 2), max(E_tmp, [], 2)]; % sort the endpoints
            [~, SG.E_sg2sp] = ismember(SG.E, E_tmp, 'rows');
        end
        
        
        function dmax = compute_max_dist(SG, vid1, vid2)
            % compute the maximum distance between two nodes
            % it is constrainted by the input smocking pattern
            % assume the fabric cannot be stretched
            
            % the corresponding grid vtx from the input cloth
            i_vid_sp = SG.vid_from_sp{vid1};
            j_vid_sp = SG.vid_from_sp{vid2};
            % find the pairwise distance between the vtx
            d = pdist2(SG.SP.V(i_vid_sp, :), SG.SP.V(j_vid_sp, :), 'euclidean');
            dmax = min(d(:));
        end
        
        function emax = compute_max_edge_length(SG, eid)
            % compute the maximum edge length
            emax = SG.compute_max_dist(SG.E(eid,1), SG.E(eid, 2));
        end
        
        
        function [] = plot(obj, mode)
            if mode == 1 % plot the complete smocked graph
                for i = reshape(obj.eid_pleat, 1, [])
                    plot(obj.V(obj.E(i,:), 1), obj.V(obj.E(i,:), 2), 'k'); hold on;
                end
                for i = reshape(obj.eid_underlay, 1, [])
                    plot(obj.V(obj.E(i,:), 1), obj.V(obj.E(i,:), 2), 'r'); hold on;
                end
                
                scatter(obj.V(obj.vid_underlay,1), obj.V(obj.vid_underlay,2), 'filled','r'); hold on;
                scatter(obj.V(obj.vid_pleat,1), obj.V(obj.vid_pleat,2), 'filled','k'); hold on;
                
                for i = 1:length(obj.vid_underlay)
                    vid = obj.vid_underlay(i);
                    text(obj.V(vid, 1), obj.V(vid, 2), num2str(vid));
                end
                
                axis equal; axis off;
                view([0,90])
                set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
                set(gca, 'YTIck', 0:2:obj.SP.len_y);
                set(gca, 'XTick', 2:2:obj.SP.len_x)
            end
        end
    end
end