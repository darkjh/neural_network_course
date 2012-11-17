function data=createData(Nc,range,varC,sizeC,dim)
%createData   Creates random data in form of clusters.
%
% data=createData(Nc,dim,varC,sizeC)
%
%  Input and output arguments: 
%   data     (matrix) output data. Size= samples X dimension (in this case 2)
%
%   Nc       (scalar) desired cluster count
%   range    (scalar) a rough estimation of the maximal values of the
%                     random data
%   varC     (scalar) describes how big the cluster is allowed to get
%   sizeC    (scalar) describes how many datapoints there are per cluster
%   dim      (scalar) dimensionality of the datapoints
data=zeros(sizeC*Nc,dim);
c=1;
for i=1:Nc
    C=rand(1,dim)*range;
    for j=1:sizeC
        data(c,:)=C+rand(1,dim)*(varC/2)-varC;
        c=c+1;
    end
end