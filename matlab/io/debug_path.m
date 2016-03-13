function output = debug_path(p)
    global debug_count;

    output = sprintf('%02d_%s', debug_count,p);
    debug_count = debug_count + 1;
end % debug_path
