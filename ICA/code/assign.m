function res = assign(imgsRecons, imgsOrigin)
    [N,~] = size(imgsRecons);
    max = 0;
    res = ones(N,1);
    
    % find the most correlated image to each w vector %
    for i = 1:N
        for j = 1: N
            temp = corrcoef(imgsRecons(i,:), imgsOrigin(j,:));
            if( abs(temp(1,2)) > max)
                max = abs(temp(1,2));
                index = j;
            end
        end
        res(i) = index;
        max = 0;
    end
   
end

