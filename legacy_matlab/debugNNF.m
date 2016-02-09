
frame = 10;
figure;
imagesc(nnf(:,:,frame,3));

[h,w, nF ,~] = size(nnf);

nCurves = 1;

figure;
for i = 1:nCurves
    color = rand(1,3);
    hh = randi(h);
    ww = randi(w);
    xx = squeeze(nnf(hh,ww,:,3));
    plot(xx, 'color', color)
    hold on
end
ylabel('w')
xlabel('time')

figure;
for i = 1:nCurves
    color = rand(1,3);
    hh = randi(h);
    ww = randi(w);
    xx = squeeze(nnf(hh,ww,:,1));
    plot(xx, 'color', color)
    hold on
end
ylabel('u')
xlabel('time')

figure;
for i = 1:nCurves
    color = rand(1,3);
    hh = randi(h);
    ff = randi(nF);
    xx = squeeze(nnf(hh,:,ff,1));
    plot(xx, 'color', color)
    hold on
end
ylabel('u')
xlabel('x')

figure;
for i = 1:nCurves
    color = rand(1,3);
    hh = randi(h);
    ff = randi(nF);
    xx = squeeze(nnf(hh,:,ff,3));
    plot(xx, 'color', color)
    hold on
end
ylabel('w')
xlabel('x')
