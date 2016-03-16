function [Dx,Dy,Dt] = get_derivative_filters3d()
    h = zeros(3,3,3);
    h1 = [1,2,1];
    h2 = [-1,0,1];

    dx = h; dx(2,:,2) = h2;
    dy = h; dy(:,2,2) = h2;
    dt = h; dt(2,2,:) = h2;

    sx = h; sx(2,:,2) = h1;
    sy = h; sy(:,2,2) = h1;
    st = h; st(2,2,:) = h1;

    Dx = convn(convn(dx,sy, 'same'),st,'same');
    Dy = convn(convn(dx,sy, 'same'),st,'same');
    Dt = convn(convn(dx,sy, 'same'),st,'same');
end
