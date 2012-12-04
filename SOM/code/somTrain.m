function error = somTrain(data, sigma, labels, sizeK)
display(sigma)
display(sizeK)

dim = 28*28; % dimension of the images
range = 255; % input range of the images ([0, 255])
[dy, dx]=size(data);

centers=rand(sizeK^2,dim)*range;

% build a neighborhood matrix
neighbor = reshape(1:sizeK^2,sizeK,sizeK);

% YOU HAVE TO SET A LEARNING RATE HERE:
eta = 0.01;

tmax=2000;
testPercentage = 0.1;

iR = randperm(tmax);
iR1 = iR(1:tmax*testPercentage); %% test
iR2 = iR(tmax*testPercentage+1:2000); %% train

counter = 0;

tau = 1000 / log(sigma);

while(1)
    
    temp = centers;
    maxRound = tmax*(1-testPercentage);
    for t=1:maxRound 
        i=iR2(t);
        sgm = sigma*exp(-(t+counter*maxRound)/tau);
%         sgm = sigma;
        centers=som_step(centers,data(i,:),neighbor,eta,sgm);
    end
    counter = counter + 1;
    converge = norm(centers - temp)/norm(temp);
    display(counter)
    display(converge)
    display(sgm)
      
    if (converge < 0.01)
        break;
    end
    
 end;

%res = assigne(centers, data, labels, size(targetdigits,2))
assign = knnclassify(centers, data(iR1,:), labels(iR1), 5);
% assign = knnclassify(centers, data, labels, 5);
reshape(assign, sizeK, sizeK)'
error = errorRate(data(iR1,:), centers, assign, labels(iR1))

for i=1:sizeK^2
    subplot(sizeK,sizeK,i);
    imagesc(reshape(centers(i,:),28,28)'); colormap gray;
    axis off
end


end

