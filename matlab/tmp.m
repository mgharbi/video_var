g = init();

I = imread(fullfile(g.path.data,'rockettes01', '0001.png'));
imwrite(I, fullfile(g.path.test_data, 'rockettes01.png'));

I = imread(fullfile(g.path.data,'rockettes01', '0005.png'));
imwrite(I, fullfile(g.path.test_data, 'rockettes02.png'));
