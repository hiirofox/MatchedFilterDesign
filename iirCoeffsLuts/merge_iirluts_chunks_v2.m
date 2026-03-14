function merge_iirluts_chunks(inputDir, outputFile)
%MERGE_IIRLUTS_CHUNKS Merge chunk_*.txt files, sort by first 4 integers, and write to output.
%
% This version is robust to records that were accidentally wrapped onto
% multiple physical lines. A new record is recognized only when a line
% starts with 4 integers. Any following non-empty lines that do NOT start
% with 4 integers are treated as continuation lines and appended to the
% previous record.
%
% Usage:
%   merge_iirluts_chunks
%   merge_iirluts_chunks('iirluts_chunks', 'iirluts.txt')

    if nargin < 1 || isempty(inputDir)
        inputDir = 'iirluts_chunks';
    end
    if nargin < 2 || isempty(outputFile)
        outputFile = 'iirluts.txt';
    end

    files = dir(fullfile(inputDir, 'chunk_*.txt'));
    if isempty(files)
        error('No chunk_*.txt files found in folder: %s', inputDir);
    end

    % Sort files by name to make processing deterministic.
    [~, order] = sort({files.name});
    files = files(order);

    allKeys = zeros(0, 4);
    allLines = strings(0, 1);

    totalWrapped = 0;
    totalMalformed = 0;

    for f = 1:numel(files)
        filePath = fullfile(files(f).folder, files(f).name);
        fid = fopen(filePath, 'r');
        if fid == -1
            error('Cannot open file: %s', filePath);
        end

        cleaner = onCleanup(@() fclose(fid));

        currentRecord = "";
        currentKey = [];
        lineNum = 0;
        wrappedCount = 0;
        malformedCount = 0;

        while true
            tline = fgetl(fid);
            if ~ischar(tline)
                break;
            end
            lineNum = lineNum + 1;
            s = strtrim(string(tline));

            if strlength(s) == 0
                continue;
            end

            key = parseLeading4Ints(s);

            if ~isempty(key)
                % Flush previous record first.
                if strlength(currentRecord) > 0
                    allKeys(end+1, :) = currentKey; %#ok<AGROW>
                    allLines(end+1, 1) = currentRecord; %#ok<AGROW>
                end

                currentRecord = s;
                currentKey = key;
            else
                % Continuation line: append to previous record if one exists.
                if strlength(currentRecord) > 0
                    currentRecord = currentRecord + " " + s;
                    wrappedCount = wrappedCount + 1;
                else
                    malformedCount = malformedCount + 1;
                    warning('Skipping malformed line with no active record: %s (line %d)', filePath, lineNum);
                end
            end
        end

        % Flush last record in this file.
        if strlength(currentRecord) > 0
            allKeys(end+1, :) = currentKey; %#ok<AGROW>
            allLines(end+1, 1) = currentRecord; %#ok<AGROW>
        end

        clear cleaner;

        totalWrapped = totalWrapped + wrappedCount;
        totalMalformed = totalMalformed + malformedCount;

        fprintf('Processed %s: %d wrapped continuation lines merged, %d malformed orphan lines skipped.\n', ...
            files(f).name, wrappedCount, malformedCount);
    end

    if isempty(allLines)
        error('No valid records were parsed.');
    end

    % Sort lexicographically by the 4 dimensions.
    sortMat = [allKeys, (1:size(allKeys,1)).'];
    sortMat = sortrows(sortMat, 1:4);
    sortedIdx = sortMat(:, 5);

    sortedKeys = allKeys(sortedIdx, :); %#ok<NASGU>
    sortedLines = allLines(sortedIdx);

    % Check duplicates.
    [uniqueKeys, ia, ic] = unique(allKeys, 'rows', 'stable'); %#ok<ASGLU>
    if numel(ia) < size(allKeys, 1)
        counts = accumarray(ic, 1);
        dupRows = find(counts > 1);
        warning('Found %d duplicated 4D indices.', numel(dupRows));
        for k = 1:min(numel(dupRows), 20)
            idx = dupRows(k);
            fprintf('Duplicate key: [%d %d %d %d], count=%d\n', uniqueKeys(idx, :), counts(idx));
        end
        if numel(dupRows) > 20
            fprintf('... and %d more duplicated keys.\n', numel(dupRows) - 20);
        end
    end

    % Write output.
    fout = fopen(outputFile, 'w');
    if fout == -1
        error('Cannot open output file for writing: %s', outputFile);
    end
    cleanerOut = onCleanup(@() fclose(fout));

    for i = 1:numel(sortedLines)
        fprintf(fout, '%s\n', sortedLines(i));
    end

    clear cleanerOut;

    fprintf('Done. Wrote %d records to %s\n', numel(sortedLines), outputFile);
    fprintf('Total merged wrapped lines: %d\n', totalWrapped);
    fprintf('Total skipped orphan malformed lines: %d\n', totalMalformed);
end

function key = parseLeading4Ints(s)
% Return [i1 i2 i3 i4] if line starts with 4 integers, else [].
    tok = regexp(char(s), '^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)(?=\s|$)', 'tokens', 'once');
    if isempty(tok)
        key = [];
    else
        key = [str2double(tok{1}), str2double(tok{2}), str2double(tok{3}), str2double(tok{4})];
    end
end
