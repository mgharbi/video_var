function particles = loadParticleVideo(path)
    f = fopen(path,'r');

    l = fgetl(f);
    n = str2double(l);

    particles.count = n;
    particles.p = cell(n,1);
    particles.minLife = Inf;
    particles.maxLife = 0;

    for i = 1:n
        l = fgetl(f);
        parsed = parseLine(l);
        particles.p{i} = parsed;
        particles.minLife = min(particles.minLife,parsed.end-parsed.start+1);
        particles.maxLife = max(particles.maxLife,parsed.end-parsed.start+1);
    end
end % loadParticleVideo

%% TODO: treat the inactive particle case
function parsed = parseLine(l)
    desc = str2num(l);
    parsed.start = desc(1); 
    parsed.end = desc(2); 
    n = parsed.start - parsed.end+1;
    parsed.x = zeros(1,n);
    parsed.y = zeros(1,n);
    parsed.t = zeros(1,n);
    part_to_go = 0;
    part = 1;
    i = 3;
    while i <= length(desc)
        if part_to_go == 0
            part_to_go = desc(i);
            i = i+1;
        else
            part_to_go = part_to_go - 1;
            parsed.x(part) = desc(i);
            parsed.y(part) = desc(i+1);
            i = i +2;
            part = part + 1;
        end
    end
end
