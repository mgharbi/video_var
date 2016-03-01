function globals = init()
    root                   = mfilename('fullpath');
    [root,~,~]             = fileparts(root);
    [root,~,~]             = fileparts(root);

    % Define directories
    globals.path.root        = root;
    globals.path.data        = fullfile(globals.path.root,'data');
    globals.path.output      = fullfile(globals.path.root,'output');
    globals.path.test        = fullfile(globals.path.root,'matlab', 'test');
    globals.path.test_data   = fullfile(globals.path.test,'test_data');
    globals.path.test_output = fullfile(globals.path.output,'test_output');

    % Create directories
    for fn = fieldnames(globals.path)'
        p = globals.path.(fn{1});
        if ~exist(p,'dir')
            mkdir(p);
        end
    end

    % Add to matlab path
    addpath('../lib/mex');
    addpath('viz');
    addpath('io');
    addpath('interpolate');
    addpath('nlvv');
    addpath('test');
    addpath('params');
    addpath('synth');
end % init
