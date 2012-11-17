% xw:  the whitened data
% w:   the initial w vector
% g:   the function g
% gd:  the derivative of g
% tol: the error tolerance

function res = findSingleIC( xw, w, g, gd, tol)

    [~, N]= size(xw);
    wNew = (1/N)*sum(xw .* [g(w.'*xw) ; g(w.'*xw)], 2)- w * (1/N * sum(gd(w.'*xw)));
    wNew = wNew ./ norm(wNew);
    
    counter = 1; 
    while(norm(abs(wNew) - abs(w)) / norm(w) > tol)
    %while(~isConverge(wNew, w, tol))
        w = wNew ;
        wNew = (1/N)*sum(xw .* [g(w.'*xw) ; g(w.'*xw)], 2)- w * (1/N * sum(gd(w.'*xw)));
        wNew = wNew ./ norm(wNew);
        counter = counter + 1;  
     
        %display(wNew);
    end
        display(counter);
    res = wNew;
end


