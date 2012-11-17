function data=miximages8
x = double(imread('i1.gif')); source(1,:)=x(:);
x = double(imread('i2.gif')); source(2,:)=x(:);
x = double(imread('i3.gif')); source(3,:)=x(:);
x = double(imread('i4.gif')); source(4,:)=x(:);
x = double(imread('i5.gif')); source(5,:)=x(:);
x = double(imread('i6.gif')); source(6,:)=x(:);
x = double(imread('i7.gif')); source(7,:)=x(:);
x = double(imread('i8.gif')); source(8,:)=x(:);
mix=rand(8,8);
data=mix*source;
