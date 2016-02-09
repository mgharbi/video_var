function proj = projectAffine(pos, A)
    X1 = pos(:,1);
    Y1 = pos(:,2);
    T1 = pos(:,3);
    projX2 = A(1,1)*X1 + A(1,2)*Y1 + A(1,3)*T1 + A(1,4);
    projY2 = A(2,1)*X1 + A(2,2)*Y1 + A(2,3)*T1 + A(2,4);
    projT2 = A(3,1)*X1 + A(3,2)*Y1 + A(3,3)*T1 + A(3,4);
    proj = cat(2,projX2, projY2, projT2);
end % projectAffine
