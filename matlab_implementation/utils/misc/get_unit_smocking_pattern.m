% get the basic smocking patter
% [base_x, base_y]: the dimension of the unit smocking pattern
% i.e., we create a grid [0, base_x] X [0, base_y] with stepsize 1
%
% the sewingLines stores a list of sewing lines
% each sewing line is [p1x, p1y; p2x, p2y; p3x, p3y...]
% i.e., the xy coord of p1, p2, p3, which are supposed to be stitched
% together later
function [base_x, base_y,  sewingLines] = get_unit_smocking_pattern(type)

switch lower(type)
    case 'basket' % basket woven
        base_x = 2;
        base_y = 2;
        sewingLines = {[1,0; 0,1], ...
            [1, 1;  2, 2]};
        
    case 'zigzag' % zigzag wave
        base_x = 4;
        base_y = 2;
        sewingLines = {[1,0; 0, 1],...
            [2,0; 3,1]};
        
    case 'bone'
        base_x = 2;
        base_y = 2;
        sewingLines = {[0, 0; 1, 1]};
        
    case 'braid' % compared to the basket woven pattern, it has one empty column
        base_x = 3;
        base_y = 2;
        sewingLines = {[1,0; 0,1], ...
            [1, 1; 2, 2]};
        
    case 'heart'
        base_x = 4;
        base_y = 3;
        sewingLines = {[1,0; 0, 1; 1, 1; 0, 2], ...
            [2,0; 3,1; 2,1; 3, 2]};
        
        
    case 'arrow'
        base_x = 4;
        base_y = 2;
        sewingLines = {[0,0;1,1; 2,0], ...
            [2,1; 3,2;4,1]};
        
    case 'leaf'
        base_x = 6;
        base_y = 2;
        sewingLines = {[0,1; 1,2], ...
            [1,1; 2,0],...
            [3,2; 4,1],...
            [4,0; 5,1]};
       
    case 'box'
        base_x = 4;
        base_y = 6;
        sewingLines = {[1,0; 0,1], ...
            [1,1; 0,2], ...
            [2,0; 3,1],...
            [2,1; 3,2],...
            [0,3; 1,4],...
            [0,4; 1,5],...
            [2,4; 3,3],...
            [2,5; 3,4]};
       
    case 'twist'
        base_x = 4;
        base_y = 2;
        sewingLines = {[0,0; 1,1; 2,0; 3,1]};
      
        
        
    case 'brick'
        base_x =  8;
        base_y =  8;
        sewingLines =  {[0,0; 1,1; 2,0], ...
            [3,0;2,1;3,2], ...
            [3,3;2,2;1,3],...
            [0,3;1,2;0,1], ...
            [4,0;5,1;4,2],...
            [5,0;6,1;7,0],...
            [7,1;6,2;7,3],...
            [6,3;5,2;4,3],...
            [1,4;2,5;3,4],...
            [3,5;2,6;3,7],...
            [2,7;1,6;0,7],...
            [0,4;1,5;0,6],...
            [4,4;5,5;6,4],...
            [7,4;6,5;7,6],...
            [7,7;6,6;5,7],...
            [4,7;5,6;4,5]};
        
    case 's1'
        base_x = 2;
        base_y = 2;
        sewingLines = {[0,0; 1,1;], ...
            [0,1; 1,0]};
    
    case 's2'
        base_x = 6;
        base_y = 2;
        sewingLines = {[1,0; 0,1], ...
            [1,1; 2,2; 3,1], ...
            [3,0; 4,1],...
            [4,2; 5,1; 6,2]};
        
    otherwise
        error('Smocking type not defined!');
end


end
