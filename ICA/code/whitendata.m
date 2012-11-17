function wData=whitendata(data)
C=cov(data');
[v d]=eig(C);
wM = inv (sqrt (d)) * v';
wData =  wM * data;