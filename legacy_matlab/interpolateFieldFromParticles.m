function interpolated = interpolateFieldFromParticles(field, particles)
    P = [particles.p{:}];
    sz = size(field);
    [X,Y,T] = meshgrid(1:sz(2),1:sz(1),1:sz(3));
    X = squeeze(X(:));
    Y = squeeze(Y(:));
    T = squeeze(T(:));


    nVox = prod(sz(1:3));
    interpolated = zeros(prod(sz,1));
    x = [P.x]';
    y = [P.y]';
    t = [P.t]';
    u = [P.u]';
    v = [P.v]';
    w = [P.w]';

    fprintf('\r 1')
    F = scatteredInterpolant(x,y,t, u);
    interpolated(1:nVox) = F(X,Y,T);

    fprintf('\r 2')
    F = scatteredInterpolant(x,y,t, v);
    interpolated(nVox+1:2*nVox) = F(X,Y,T);

    fprintf('\r 3')
    F = scatteredInterpolant(x,y,t, w);
    interpolated(2*nVox+1:3*nVox) = F(X,Y,T);

    interpolated = reshape(interpolated,sz);
    fprintf('\n')
end % interpolateFieldFromParticles
