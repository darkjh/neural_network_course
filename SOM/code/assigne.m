function sol = assigne( centers, data, labels, nbDigit)
%ASSIGNE   assigne a digit to each center
%
%  assigne( centers, data, labels)
%
%  Input and output arguments: 
%   sol      (matrix) the digit matrix corresponding to the image matrix
%   data     (vector) the actually presented datapoint to be presented in this timestep
%   labels   (vector) the labels corresponding to the data points
%   nbDigit  (scalar) the number of Digits which will be assigned.

    % initialization
    N = size(centers, 1);
    sizeK = sqrt(N);
    P = size(data, 1);
    nbSurround = P/nbDigit;
    res = 1:N;  
    dist = 1:P;

    for i = 1:N  % for each center
        
        for j = 1:P % for each data point
            dist(j) = sqrt(sum((data(j,:)-centers(i,:)).^2,2));
        end
        
        [~,idx] = sort(dist);
        x = labels(idx(1:nbSurround));  % choose the first nbSurround data points
        x = x';
        [nb,lb]=hist(x,unique(x)); % get the distribution of these data points.
        [~, idxlb] = max(nb);
        res(i) = lb(idxlb); % the digit which has the largest occurrence
    end
    
    sol = reshape(res, sizeK, sizeK)';

end

