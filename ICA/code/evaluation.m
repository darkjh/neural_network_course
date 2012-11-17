function quality = evaluation(wNew, source, xw)

    orth = [wNew(2);  -wNew(1)];


    img(1,:) = wNew' * xw;
    img(2,:) = orth' * xw;

    % assign the reconstructed picture to the right source image %
    order = assign(img, source);

    % correspanding correlation coeffient between two images %
    if order(1) == 1 % the first reconstructed picture corresponds to the first image.
        coef = corrcoef(img(1,:), source(1,:));
        res(1) = coef(1,2);
        coef = corrcoef(img(2,:), source(2,:));
        res(2) = coef(1,2);
    else
        coef = corrcoef(img(2,:), source(1,:));
        res(1) = coef(1,2);
        coef = corrcoef(img(1,:), source(2,:));
        res(2) = coef(1,2);
    end

    %quality = sqrt(res(1)^2+res(2)^2);
    quality = mean(abs(res));
    display(quality)
    display('-------------------------------')
end

