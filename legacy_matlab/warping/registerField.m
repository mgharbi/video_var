function registered = registerField(field,flow)
    sz = size(field);
    registered = field;
    for frame = sz(3):-1:2
        for other = frame %:sz(3)
        registered(:,:,other,:) = ...
            backWarp2D(squeeze(registered(:,:,other,:)), squeeze(flow(:,:,frame-1,:)));
        end
    end
end % registerField
