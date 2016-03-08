function video = oscillating_square(sz, rect_hsz, frequency_ramp, MaxAmp, fig)


    % image dimensions
    Ny = sz(1);
    Nx = sz(2);


    % number of frames per cycle
    Nf = numel(frequency_ramp);

    img  = uint8(255*rand(2*rect_hsz(1)+1,2*rect_hsz(2)+1,3));
    bg  = uint8(zeros(Ny,Nx,1));
    % bg  = uint8(127*rand(Ny,Nx,1));
    bg = repmat(bg, [1,1,3]);


    % changing the sin freq.
    t = linspace(0,1, Nf);

    video = zeros(Ny,Nx,Nf,3, 'uint8');

    offset = MaxAmp*sin(2*pi*frequency_ramp.*t); 

    midx = Nx-1/2;
    midy = Ny-1/2;
    minx = round((Nx+1)/2 - rect_hsz(2))*ones(Nf,1);
    maxx = round((Nx+1)/2 + rect_hsz(2))*ones(Nf,1);
    miny = round((Ny+1)/2 - rect_hsz(1) + offset);
    maxy   = round((Ny+1)/2 + offset + rect_hsz(1));

    ref_frame = bg;
    ref_frame(miny(1):maxy(1), minx(1):maxx(1), :) = img;

    for f=1:Nf;
        frame = imtranslate(ref_frame,[0,offset(f)],'method', 'cubic');
        % frame = bg;
        % frame(miny(f):maxy(f), minx(f):maxx(f), :) = img;
        video(:,:,f,:) = frame;
    end

    if nargin >= 5
        tplot = 1:Nf;
        fprintf('* Making 3d plot\n');
        figure(fig);
        plot3(tplot,minx,miny, ':', 'Color', [0 0.4470 0.7410]);
        title('Oscillating square')
        axis equal;
        axis([0, Nf, 0, Nx, 0, Ny])
        set(gca,'ydir','reverse')
        hold on;
        plot3(tplot,minx,maxy, ':', 'Color', [0 0.4470 0.7410]);
        plot3(tplot,maxx,maxy, ':', 'Color', [0 0.4470 0.7410]);
        plot3(tplot,maxx,miny, ':', 'Color', [0 0.4470 0.7410]);
        for f = 1:5:Nf
            plot3([f,f,f,f,f], [minx(f),minx(f), maxx(f), maxx(f),minx(f)], [miny(f),maxy(f), maxy(f), miny(f),miny(f)], ':',  'Color', [0 0.4470 0.7410])
        end
        xlabel('t')
        ylabel('x')
        zlabel('y')
    end
end % oscillating_square
