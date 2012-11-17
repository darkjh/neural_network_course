%Example for using createData plotData and som_step
clear all
close all
clc

%initialise data with one cluster of width 5 and 30 datapoints.
Nc=1;
dim=2;
range=10;
varC=2;
sizeC=100;
data=createData(Nc,range,varC,sizeC,dim);
[dy dx]=size(data);

%set the size of the Kohonen map. In this case it will be 6 X 6
sizeK=6;

%set the width of the neighborhood via the width of the gaussian that
%describes it
sigma=sqrt(2*sizeK^2)/6;

%initialise the centers randomly
centers=rand(sizeK^2,dim)*range;

%build a neighborhood matrix
neighbor=reshape(1:sizeK^2,sizeK,sizeK);

%set the learning rate
eta=0.3;

%set the maximal iteration count
tmax=400;

%set the random order in which the datapoints should be presented
iR=mod(randperm(tmax),dy)+1;

% why we have a proper rate of simulation, not too fast?
for t=1:tmax
    i=iR(t);
    centers=som_step(centers,data(i,:),neighbor,eta,sigma);
    plotData(centers,data,neighbor);
%     pause();
end
