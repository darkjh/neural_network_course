function res = errorRate(data, centers, assign, labels)
    n = size(data, 1);
    tmp = knnclassify(data, centers, assign);
    delta = tmp - labels;
    res = size(delta(delta ~= 0), 1) / n;
end

