function pth = writeJutulInputs(G, rock, name, folder)
    if nargin < 3 || isempty(name),   name = 'gridrock'; end
    if nargin < 4 || isempty(folder), folder = fullfile(mrstOutputDirectory(), 'jutul'); end

    % If folder exists, remove it completely
    if exist(folder, 'dir')
        rmdir(folder, 's');  % Remove folder and all contents
    end
    mkdir(folder);  % Create new empty folder

    % Save only G and rock (flat, v7.3 for large data)
    pth  = fullfile(folder, [name, '.mat']);
    data = struct('name', name, 'G', G, 'rock', rock);
    save(pth, '-struct', 'data', '-v7.3');

    % Make path Julia-friendly on Windows
    pth(strfind(pth, '\')) = '/';
end


