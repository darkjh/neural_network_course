function data=miximages2
x = double(imread('i1.gif')); source(1,:)=x(:);
x = double(imread('i2.gif')); source(2,:)=x(:);
mix=rand(2,2);
data=mix*source;
