function particles = processParticles(particles)
    fsize = 1;
    for i = 1:particles.count
        p = particles.p{i};
        n = length(p.x);
        if n < 2*fsize+1
            fprintf('\r%04d/%04d',i,particles.count);
            continue
        end
        p = particles.p{i};

        particles.p{i} = p;
        fu = fft(p.u);
        fv = fft(p.v);
        fw = fft(p.w);
        l = min(fsize,floor(n/2));
        % if mod(n,2)==1
        %     fu(1) = 0;
        %     fu(2:1+l) = 0;
        %     fu(end-l+1:end) = 0;

        %     fv(1) = 0;
        %     fv(2:1+l) = 0;
        %     fv(end-l+1:end) = 0;
        %     
        %     fw(1) = 0;
        %     fw(2:1+l) = 0;
        %     fw(end-l+1:end) = 0;
        % else
            fu(1:l+1)=0;
            fu(end-l+1:end)=0;

            fv(1:l+1)=0;
            fv(end-l+1:end)=0;

            fw(1:l+1)=0;
            fw(end-l+1:end)=0;
        % end

        p.u = ifft(fu);
        p.v = ifft(fv);
        p.w = ifft(fw);

        particles.p{i} = p;


        fprintf('\r%04d/%04d',i,particles.count);
    end
    fprintf('\n');
end % processParticles
