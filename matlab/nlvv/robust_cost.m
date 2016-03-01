function cost = robust_cost(x,y)
    cost = sqrt(mean((single(x)-single(y)).^2, 4)+1e-6);
end % robust_cost
