function centers=som_step(centers,data,neighbor,eta,sigma)
%som_step   performs one step of the sequential learning for a self
%           organized map (som)
%
%  som_step(centers,data,neighbor,eta,sigma)
%
%  Input and output arguments: 
%   centers  (matrix) cluster centres. Have to be in format:
%                     center X dimension
%   data     (vector) the actually presented datapoint to be presented in
%                     this timestep
%   neighbor (matrix) the coordinates of the centers in the desired
%                     neighborhood.
%   eta      (scalar) a learning rate
%   sigma    (scalar) the width of the gaussian neighborhood function.
sizeK = sqrt(size(centers,1));

%find the best matching unit via the minimal distance to the datapoint
[~, winningunit]=min(sqrt(sum(abs(centers-repmat(data,sizeK^2,1)).^2,2)));

%find coordinates of the winner
[a, b]=find(neighbor==winningunit);

for j=1:sizeK^2 % update all units weighted by their distance to the winner
    [a1, b1]=find(neighbor==j); % find the coordinates of unit j
    
    disc = exp(-((a-a1)^2+(b-b1)^2)/(2*sigma^2)); % calculate distance
    
    % update winner and neighbors according to the neighborhood function    
    centers(j,:)=centers(j,:)+disc*eta*(data-centers(j,:));
end

