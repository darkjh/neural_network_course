function K = knn_learning(trainning, labels)

mdl = ClassificationKNN.fit(trainning,labels);
loss = 1;

for k = 1:1:10
    display(k);
    mdl.NumNeighbors = k;
    cvmdl = crossval(mdl,'kfold',10);
    kloss = kfoldLoss(cvmdl);
    if kloss < loss
        loss = kloss;
        K = k;
    end
end

end