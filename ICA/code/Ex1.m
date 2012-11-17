format long

% data source %
tmp = double(imread('i1.gif')); source(1,:)=tmp(:);
tmp = double(imread('i2.gif')); source(2,:)=tmp(:);

x = miximages2;

xCentralized = centre(x);

xw = whitendata(xCentralized); 

% data points figure %

%figure(1)
%plot(source(1,:), source(2,:))

%figure(2)
%plot(x(1,:), x(2,:))

%figure(3)
%plot(xw(1,:), xw(2,:))


% initial w %
w = [1;0];

g1 = inline('tanh(y)','y');
g1d = inline('sech(y).^2', 'y');

g2 = inline('y.*exp(-y.^2/2)','y');
g2d = inline('exp(-y.^2/2)-y.^2.*exp(-y.^2/2)','y');

g3 = inline('y.^3','y');
g3d = inline('3.*y.^2','y');

% error tolerance %
tol = 0.000001;
%tol = eps;

wNew = findSingleIC(xw, w, g1, g1d, tol);

orth = [wNew(2);  -wNew(1)];

% Visualization %

img(1,:) = wNew' * xw;
img(2,:) = orth' * xw;

visual(img(1,:) * 256, 1)
visual(img(2,:) * 256, 2)

% % % % % % % % % % % % % % % % % % % % % % 

% Compare the three functions %

evaluation(wNew, source, xw);

wNew = findSingleIC(xw, w, g2, g2d, tol);
evaluation(wNew, source, xw);

wNew = findSingleIC(xw, w, g3, g3d, tol);
evaluation(wNew, source, xw);

    

