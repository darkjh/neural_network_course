format long

x = miximages8;

xCentralized = centre(x);

xw = whitendata(xCentralized); 

W = eye(8);


g1 = inline('tanh(y)','y');
g1d = inline('sech(y).^2', 'y');

g2 = inline('y.*exp(-y.^2/2)','y');
g2d = inline('exp(-y.^2/2)-y.^2.*exp(-y.^2/2)','y');

g3 = inline('y.^3','y');
g3d = inline('3.*y.^2','y');

tol = 0.000001;

[~, P]= size(xw); %number of points
[~, N] = size(W); %number of input channel


W = findAllICs(xw, W, g1, g1d, tol);


% Visualization %

imageRecons = W' * xw;

% stock the origin images in matrix source
x = double(imread('i1.gif')); source(1,:)=x(:);
x = double(imread('i2.gif')); source(2,:)=x(:);
x = double(imread('i3.gif')); source(3,:)=x(:);
x = double(imread('i4.gif')); source(4,:)=x(:);
x = double(imread('i5.gif')); source(5,:)=x(:);
x = double(imread('i6.gif')); source(6,:)=x(:);
x = double(imread('i7.gif')); source(7,:)=x(:);
x = double(imread('i8.gif')); source(8,:)=x(:);


for i = 1 : 8
    visual(imageRecons(i,:)*256, i)
end
	
display(assign(imageRecons, source))



