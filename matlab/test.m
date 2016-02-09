addpath('../lib/mex');

A = uint8(ones(10,20,5,3));
B = uint8(255*rand(10,20,5,3));
Y = stwarp(A, B);
size(Y)

clear Y;

