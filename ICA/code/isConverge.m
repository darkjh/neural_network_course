function res = isConverge(WNew, W, tol)
    [~,N] = size(W);
    res = 1;
    for i = 1 : N
        if (norm(abs(WNew(:,i)) - abs(W(:,i)))/norm(W(:,i)) > tol)
            res = 0;
        end
    end
end