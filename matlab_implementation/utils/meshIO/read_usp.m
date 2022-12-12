function usp = read_usp(filename)
fp = fopen(filename, 'r');
type = fscanf( fp, '%s', 1 );
sewingLines = cell(1, 10);


while strcmp( type, '' ) == 0
    line = fgets(fp);
    if strcmp(type, 'gridX') == 1
        base_x = sscanf(line, '%d');
    end

    if strcmp(type, 'gridY') == 1
        base_y = sscanf(line, '%d');
    end

    if strcmp(type, 'sp') == 1
        sl = sscanf(line, '%d %f %f %f');
        if sl(1)+1 > length(sewingLines)
            sewingLines = [sewingLines, cell(1,10)];
        end
        sewingLines{sl(1)+1} = [sewingLines{sl(1)+1}; reshape(sl(2:3),1,[])];

        num_sl = sl(1)+1;
    end

    type = fscanf( fp, '%s', 1);
end

sewingLines = sewingLines(1:num_sl);

usp = struct();
usp.base_x = base_x;
usp.base_y = base_y;
usp.sewingLines = sewingLines;

tmp = strsplit(filename, '/');
name = tmp{end};
usp.name = name(1:end-4);

end