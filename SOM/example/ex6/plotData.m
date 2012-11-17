function plotData(centers, data, neighbor)
%plotData   plots som data. This includes the used datapoints as well as
%           the cluster centres and neighborhoods.
%
%  plotData(centers, data, neighbor)
%
%  Input and output arguments: 
%   centers  (matrix) cluster centres to be plotted. Have to be in format:
%                     center X dimension (in this case 2)
%   data     (matrix) datapoints to be plotted. have to be in the same
%                     format as centers
%   neighbor (matrix) the coordinates of the centers in the desired
%                     neighborhood.
[a b]=size(centers);
figure(1)
hold off

%plot centers and datapoints
if b==3
    scatter3(centers(:,1),centers(:,2),centers(:,3),'rx')
    hold on;
    scatter3(data(:,1),data(:,2),data(:,3),'bo')
    hold on
else
    scatter(centers(:,1),centers(:,2),'rx')
    hold on;
    scatter(data(:,1),data(:,2),'bo')
    hold on
end

%plot neighborhood grid

%figure(2)
if b==3
    for g=1:length(neighbor)
        plot3(centers(neighbor(g,:),1),centers(neighbor(g,:),2),centers(neighbor(g,:),3),'k')
        hold on
    end
    for g=1:length(neighbor)
        plot3(centers(neighbor(:,g),1),centers(neighbor(:,g),2),centers(neighbor(:,g),3),'k')
        hold on 
   end
    %drawnow;
else
    for g=1:length(neighbor)
        plot(centers(neighbor(g,:),1),centers(neighbor(g,:),2),'k')
        hold on
    end
    for g=1:length(neighbor)
        plot(centers(neighbor(:,g),1),centers(neighbor(:,g),2),'k')
        hold on
    end
    %drawnow;
end