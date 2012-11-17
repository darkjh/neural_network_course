% Example script for basic Kohonen map algorithm.

clear all
close all

data = dlmread('data.txt'); % read in data
labels = dlmread('labels.txt'); % read in labels

name = 'ju han'; % REPLACE BY YOUR OWN NAME
targetdigits = name2digits(name); % assign the four digits that should be used

digitstoremove = setdiff(0:9,targetdigits); % the other 6 digits are removed from the data set.
for i=1:6
    data(labels==digitstoremove(i),:) = [];
    labels(labels==digitstoremove(i)) = [];
end

dim = 28*28; % dimension of the images
range = 255; % input range of the images ([0, 255])
[dy, dx]=size(data);

% set the size of the Kohonen map. In this case it will be 6 X 6
sizeK=6;

%set the width of the neighborhood via the width of the gaussian that
%describes it
sigma=1;

%initialise the centers randomly
centers=rand(sizeK^2,dim)*range;

% build a neighborhood matrix
neighbor = reshape(1:sizeK^2,sizeK,sizeK);

% YOU HAVE TO SET A LEARNING RATE
eta = 0.3;

%set the maximal iteration count
tmax=5000; % this might or might not work; use your own convergence criterion

%set the random order in which the datapoints should be presented
iR=mod(randperm(tmax),dy)+1;

for t=1:tmax
    i=iR(t);
    centers=som_step(centers,data(i,:),neighbor,eta,sigma);
end

%% for visualization, you can use this:
for i=1:sizeK^2
    subplot(sizeK,sizeK,i);
    imagesc(reshape(centers(i,:),28,28)'); colormap gray;
    axis off
end