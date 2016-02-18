function globals = init()
    root = mfilename('fullpath');
    [root,~,~]      = fileparts(root);
    [root,~,~]      = fileparts(root);
    globals.path.root = root;
    addpath('../lib/mex');
    addpath('viz_warp');
    addpath('io');
    addpath('interpolate');
    addpath('nlvv');
end % init
