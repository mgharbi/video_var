function video = oscillating_square(sz, rect_hsz, cycles)
    % cycles, changing the frequency between 1Hz-3Hz e.g. cycles = 1:3;

    % image dimensions
    Nx = sz;
    Ny = sz;

    % number of frames per cycle
    Nf = 30;

    % maximun ampliditue of the temporal sin
    MaxAmp = 10;

    img  = uint8(255*rand(2*rect_hsz+1,2*rect_hsz+1,3));


    % changing the sin freq.
    t = linspace(0,1, Nf+1);
    t

    video = zeros(Nx,Ny,length(cycles)*Nf,3, 'uint8');

    f = 1;
    for j=1:length(cycles)
        V{j} = MaxAmp*sin(2*pi*t*cycles(j));
        for i=1:Nf;
            start_j = round((Ny-1)/2 - rect_hsz+V{j}(i));
            end_j   = round((Ny-1)/2 + V{j}(i)+ rect_hsz);
            start_i = round((Ny-1)/2 - rect_hsz);
            end_i   = round((Ny-1)/2 + rect_hsz);

            frame = zeros(Nx, Ny,3);
            frame(start_j:end_j, start_i:end_i, :) = img;

            video(:,:,f,:) = frame;
            f = f+1;
        end
    end
end % oscillating_square
