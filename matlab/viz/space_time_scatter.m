function space_time_scatter(fig, data, w, color)
% fig is a figure handle
% data Nx3 matrix of x,y,t coordinates
    figure(fig);
    if nargin < 4
        color = [0 0.4470 0.7410];
    end
    if nargin > 2
        w = double(w(:));
        data = double(data);
        labels = num2str(w, '%.2f');
        text(data(:, 3)+1, data(:, 2)+1, data(:, 1)+1,labels)
        s = max(w*80,10);
    else
        s = 80*ones(size(data,1),1);
    end
    scatter3(data(:, 3), data(:, 2), data(:, 1), s, 'filled', 'MarkerFaceColor', color, 'MarkerEdgeColor', color);
end % space_time_scatter
