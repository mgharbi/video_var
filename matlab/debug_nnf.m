close all;
clear all;

globals = init();

res = load(fullfile(globals.path.output, 'result.mat'));
res = res.res;

[h,w,nC,nF] = size(res.video);
x_coord = floor(w/2);

% Interactive figure

nnf     = res.iter{1}{1}.nnf;
w     = res.iter{1}{1}.w;
db      = res.iter{1}{1}.regular_video;
regular = res.iter{1}{1}.new_regular_video;

before_yt = video_slice(db,2,x_coord);
after_yt  = video_slice(regular,2,x_coord);
nnf_yt    = video_slice(nnf,2,x_coord);

knn = size(nnf,4)/3;
psz = res.params.knnf.patch_size_space;
psz_t = res.params.knnf.patch_size_time;

figure;
subplot(2,knn,1:knn)
imshow(before_yt);
axis on
hold on

for k = 1:knn
    subplot(2,knn,knn+k);
    imshow(zeros(psz,psz_t,3));
end

f = 0;
while true
    [t,y] = ginput(1);
    t = round(t);
    y = round(y);
    try
        delete(f)
    end
    try
        for k = 1:knn
            delete(f2{k})
        end
    end
    nn = squeeze(nnf(y,x_coord,t,:))+1;
    xx = nn(1:3:end);
    yy = nn(2:3:end);
    tt = nn(3:3:end);
    % f2 = scatter(tt,yy,'+','b', 'linewidth', 2);
    subplot(2,knn,1:knn)
    for k = 1:knn
        f2{k} = rectangle('Position', [tt(k),yy(k),psz_t,psz], 'EdgeColor', 'blue', 'LineWidth', 2);
    end
    for k = 1:knn
        wght = w(y,x_coord,t,k);
        subplot(2,knn,knn+k)
        p = squeeze(db(yy(k):yy(k)+psz, xx(k), tt(k):tt(k)+psz_t,:));
        title(sprintf('weight: %f', wght))
        imshow(p)
    end

    subplot(2,knn,1:knn)
    f = rectangle('Position', [t,y,psz_t,psz], 'EdgeColor', 'green', 'LineWidth', 2, 'Curvature', 1);
end
