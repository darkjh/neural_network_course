function res = findAllICs(xw, W, g, gd, tol)
	[~, P]= size(xw); %number of points
	[~, N] = size(W); %number of input channle 
    
    WNew = eye(N);
    
    for i = 1:  N
		WNew(:,i) = (1/P)*sum(xw .* repmat(g(W(:,i).'*xw), N, 1), 2)- W(:,i) * (1/P * sum(gd(W(:,i)'*xw)));
		WNew(:,i) = WNew(:,i) ./ norm(WNew(:,i));
    end
      
    counter = 1; 
    while (~isConverge(WNew, W, tol))
               
        W = WNew;
        
        for i = 1:  N
            WNew(:,i) = (1/P)*sum(xw .* repmat(g(W(:,i).'*xw),  N, 1), 2)- W(:,i) * (1/P * sum(gd(W(:,i)'*xw)));
            WNew(:,i) = WNew(:,i) ./ norm(WNew(:,i));
        end
		
		WNew = (WNew*WNew')^(-0.5) * WNew; % symmetric orthogonalization
        
        counter = counter + 1; 
        %display(WNew)
    end
    
    display(counter)
    res = W;
end



