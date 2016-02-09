function particles = attachFieldToParticles(field, particles)
    s = size(field,1);
    for i = 1:particles.count
        p = particles.p{i};
        n = p.end-p.start+1;
        p.u = zeros(1,n);
        p.v = zeros(1,n);
        p.w = zeros(1,n);

        p.y = s - (p.y);

        x = round(p.x);
        y = round(p.y);
        t = (p.start+1):(p.end + 1);
        p.t = t;
        for k = 1:length(t)
            p.u(k) = field(y(k),x(k),t(k),1);
            p.v(k) = field(y(k),x(k),t(k),2);
            p.w(k) = field(y(k),x(k),t(k),3);
        end
        particles.p{i} = p;

        fprintf('\r%04d/%04d',i,particles.count);
    end
    fprintf('\n');
    % code
end % attachFieldToParticles
