function video = oscillating_square(sz, rect_hsz, cycles)
    % cycles, changing the frequency between 1Hz-3Hz e.g. cycles = 1:3;

    % image dimensions
    Nx = sz;
    Ny = sz;

    % number of frames per cycle
    Nf = 30;

    % maximun ampliditue of the temporal sin
    MaxAmp = 10;


    % changing the sin freq.
    t = linspace(0,2*pi, Nf+1);

    video = zeros(Nx,Ny,length(cycles)*Nf,3, 'uint8');

    f = 1;
    for j=1:length(cycles)
        V{j} = MaxAmp*sin(t*cycles(j));
        for i=1:Nf;
            start_j = round((Ny-1)/2 - rect_hsz+V{j}(i));
            end_j   = round((Ny-1)/2 + V{j}(i)+ rect_hsz);
            start_i = round((Ny-1)/2 - rect_hsz);
            end_i   = round((Ny-1)/2 + rect_hsz);

            frame = zeros(Nx, Ny);
            frame(start_j:end_j, start_i:end_i) = 0.5;

            video(:,:,f,1) = frame;
            video(:,:,f,2) = frame;
            video(:,:,f,3) = frame;
            f = f+1;
        end
    end
end % oscillating_square
