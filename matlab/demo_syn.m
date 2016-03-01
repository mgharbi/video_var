%%
clear all

% image dimentions
filename = ['Vid/SinFreqChangex3']
Nx = 101;
Ny = 101;
rect_hsz = 11;

% number of frames per cycle
Nf = 30
% maximun ampliditue of the temporal sin
MaxAmp = 20;

frame = zeros(Nx, Ny);
frame((Ny-1)/2 - rect_hsz:(Ny-1)/2 + rect_hsz, (Ny-1)/2 - rect_hsz:(Ny-1)/2 + rect_hsz) = 0.5;
% fimshow(frame)
% changing the sin freq.

t = linspace(0,2*pi, Nf+1);
% chancing the frequency between 1Hz-3Hz
c = 1:3
f = 1;
for j=1:length(c)
    V{j} = MaxAmp*sin(t*c(j));
    for i=1:Nf;
        frame_i = zeros(Nx, Ny);
        frame_i((Ny-1)/2 - rect_hsz+V{j}(i):(Ny-1)/2 +V{j}(i)+ rect_hsz, (Ny-1)/2 - rect_hsz:(Ny-1)/2 + rect_hsz) = 0.5;
        mov(f).cdata = im2uint8(frame_i);
        mov(i).colormap = 'gray';
        f = f+1;
    end
end

%
% writerObj = VideoWriter(filename, 'MPEG-4'); %'Uncompressed AVI'
%     open(writerObj);
%     for i = 1:length(mov)
%         writeVideo(writerObj,mov(i).cdata);
%     end
%     close(writerObj);
%     
