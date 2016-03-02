function video = oscillating_square(sz, rect_hsz, frequency_ramp)

    % image dimensions
    Nx = sz;
    Ny = sz;

    % number of frames per cycle
    Nf = numel(frequency_ramp);

    % maximun ampliditue of the temporal sin
    MaxAmp = 10;

    img  = uint8(255*rand(2*rect_hsz+1,2*rect_hsz+1,3));
    bg  = uint8(127*rand(Nx,Ny,1));
    bg = repmat(bg, [1,1,3]);


    % changing the sin freq.
    t = linspace(0,1, Nf);

    video = zeros(Nx,Ny,Nf,3, 'uint8');

    offset = MaxAmp*sin(2*pi*frequency_ramp.*t); 

    for f=1:Nf;
        start_j = round((Ny-1)/2 - rect_hsz + offset(f));
        end_j   = round((Ny-1)/2 + offset(f) + rect_hsz);
        start_i = round((Ny-1)/2 - rect_hsz);
        end_i   = round((Ny-1)/2 + rect_hsz);

        frame = bg;
        frame(start_j:end_j, start_i:end_i, :) = img;

        video(:,:,f,:) = frame;
    end
end % oscillating_square
