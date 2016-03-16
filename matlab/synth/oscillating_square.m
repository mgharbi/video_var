function video = oscillating_square(sz, rect_hsz, frequency_ramp, MaxAmp, fig)
    % image dimensions
    Ny = sz(1);
    Nx = sz(2);

    % number of frames per cycle
    Nf = numel(frequency_ramp);

    img  = single(rand(2*rect_hsz(1)+1,2*rect_hsz(2)+1,3));
    % img(:,:,1) = 1;
    % img(:,:,2) = .1;
    % img(:,:,3) = .1;
    bg  = single(zeros(Ny,Nx,1));
    bg = repmat(bg, [1,1,3]);

    bg_cube = single(zeros(Ny,Nx,1,1)); 
    bg_cube = repmat(bg_cube, [1,1,Nf,3]);


    % changing the sin freq.
    t = 1:Nf;
    n_cycles = 5;

    video = zeros(Ny,Nx,Nf,3, 'single');

    t_start = 15;
    circular_coordinate = frequency_ramp.*(t-t_start);

    t_stop = find(circular_coordinate>=n_cycles,1);
    circular_coordinate(circular_coordinate>n_cycles) = 0;
    circular_coordinate(1:t_start) = 0;
    circular_coordinate = 2*pi*circular_coordinate;

    offset = MaxAmp*profile(circular_coordinate); 

    % offset = MaxAmp*sin(2*pi*frequency_ramp.*t); 

    midx = Nx-1/2;
    midy = Ny-1/2;
    minx = round((Nx+1)/2 - rect_hsz(2))*ones(Nf,1);
    maxx = round((Nx+1)/2 + rect_hsz(2))*ones(Nf,1);
    miny = round((Ny+1)/2 - rect_hsz(1) + offset);
    maxy = round((Ny+1)/2 + offset + rect_hsz(1));

    ref_frame = bg;
    ref_frame(miny(1):maxy(1), minx(1):maxx(1), :) = img;

    for f=t_start:t_stop;
        frame = imtranslate(ref_frame,[0,offset(f)],'method', 'linear');
        % frame = bg;
        % frame(miny(f):maxy(f), minx(f):maxx(f), :) = img;
        video(:,:,f,:) = frame;
    end

    video(video==0) = bg_cube(video == 0);

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

function ret = profile(t)
    t = mod(t,2*pi);
    mode = 'triangle';
    if strcmp(mode,'triangle')
        ret = zeros(numel(t), 1);
        ret(t<pi) = (1-t(t<pi)/(pi));
        ret(t>=pi) = (t(t>=pi)-pi)/(pi);
        ret = ret*2 - 1.5;
    elseif strcmp(mode,'square')
        ret(t<pi) = -1;
        ret(t>=pi) = 1;
    end
end

