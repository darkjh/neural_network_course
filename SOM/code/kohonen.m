% Example script for basic Kohonen map algorithm.
clear all
close all

data = dlmread('data.txt'); % read in data
labels = dlmread('labels.txt'); % read in labels

name = 'ju ren'; % REPLACE BY YOUR OWN NAME
targetdigits = name2digits(name); % assign the four digits that should be used

digitstoremove = setdiff(0:9,targetdigits); % the other 6 digits are removed from the data set.
for i=1:6
    data(labels==digitstoremove(i),:) = [];
    labels(labels==digitstoremove(i)) = [];
end

% dim = 28*28; % dimension of the images
% range = 255; % input range of the images ([0, 255])
% [dy, dx]=size(data);
% 
% % set the size of the Kohonen map. In this case it will be 6 X 6
% sizeK=6;
% % sizeK=9;
% %sizeK=12;
% 
% %set the width of the neighborhood via the width of the gaussian that
% %describes it
% sigma=1;
% % sigma=3;
% %sigma=5;
% 
% %initialise the centers randomly
% centers=rand(sizeK^2,dim)*range;
% 
% % build a neighborhood matrix
% neighbor = reshape(1:sizeK^2,sizeK,sizeK);
% 
% % YOU HAVE TO SET A LEARNING RATE HERE:
% eta = 0.01;
% 
% %set the maximal iteration count
% tmax=2000; % this might or might not work; use your own convergence criterion
% 
% %set the random order in which the datapoints should be presented
% %iR=mod(randperm(tmax),dy)+1;
% iR = randperm(tmax);
% 
% % for t=1:tmax
% %     i=iR(t);
% %     centers=som_step(centers,data(i,:),neighbor,eta,sigma);
% % end

vSizeK = [6];
n = length(vSizeK);

vSigma = [1];
m = length(vSigma);

res = zeros(n, m);
for i = 1 : n
    for j = 1: m
        res(i,j) = somTrain(data, vSigma(j), labels, vSizeK(i));
    end
end

display(res);

%% for visualization, you can use this:
% for i=1:sizeK^2
%     subplot(sizeK,sizeK,i);
%     imagesc(reshape(centers(i,:),28,28)'); colormap gray;
%     axis off
% end
