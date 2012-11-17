"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb



def create_data(n_clust, data_range, var_c, size_c, dim):
    """
    Create random data in form of clusters.
    
    >>> data = create_data(n_clust,data_range,var_c,size_c, dim)
    
      Input and output arguments: 
       data     (matrix) output data. Size= samples X dimension (in this case 2)
    
       n_clust      (scalar) desired cluster count
       data_range   (scalar) a rough estimation of the maximal values of the
                         random data
       var_c        (scalar) describes how big the cluster is allowed to get
       size_c       (scalar) describes how many datapoints there are per cluster
       dim          (scalar) dimensionality of the datapoints
    """

    data = np.zeros((size_c * n_clust, dim))
    c = 0
    for i in range(n_clust):
        C = np.random.rand(1, dim) * data_range
        for j in range(size_c):
            data[c,:] = C + np.random.rand(1,dim) * (var_c/2) - var_c
            c=c+1
    return data

def kohonen():
    """Example for using create_data, plot_data and som_step.
    """
    plb.close('all')
    
    #initialise data with one cluster of width 5 and 30 datapoints.
    n_clust = 1
    dim = 2
    data_range = 10
    var_c = 4
    size_c = 100

    data = create_data(n_clust,data_range,var_c,size_c,dim)
    dy, dx = data.shape
    
    #set the size of the Kohonen map. In this case it will be 6 X 6
    size_k = 6
    
    #set the width of the neighborhood via the width of the gaussian that
    #describes it
    sigma = np.sqrt(2.0 * size_k**2)/6.0
    
    #initialise the centers randomly
    centers = np.random.rand(size_k**2, dim) * data_range
    
    #build a neighborhood matrix
    neighbor = np.arange(size_k**2).reshape((size_k, size_k))
    #set the learning rate

    eta = 0.3
    
    #set the maximal iteration count
    tmax = 400
    
    #set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)
    
    # turn "interaction" on in pylab
    plb.ion()
    # empty (for now) handles
    handles = None

    for t, i in enumerate(i_random):
        som_step(centers, data[i,:],neighbor,eta,sigma)
        handles = plot_data(centers, data, neighbor, handles)
        plb.draw()

    # leave the window open at the end of the loop
    plb.show()
    
def plot_data(centers, data, neighbor, handles):
    """Plot self-organising map (SOM) data.
    
    This includes the used datapoints as well as the cluster 
    centres and neighborhoods.
    
      plot_data(centers, data, neighbor)
    
      Input and output arguments: 
       centers  (matrix) cluster centres to be plotted. Have to be in format:
                         center X dimension (in this case 2)
       data     (matrix) datapoints to be plotted. have to be in the same
                         format as centers
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
    """

    if handles is None:
        # it is the first time we call this method: create the graphs
        a,b=centers.shape
        
        # handles = [figure, scatter_centers, scatter_data,
        #                   plot_centers_1, plot_centers_2]
        handles = []
        handles.append( plb.figure())
        
        #plot centers and datapoints
        handles.append(plb.scatter(centers[:,0],centers[:,1], c='r',marker='x', facecolor="w", edgecolor='r'))
        handles.append(plb.scatter(data[:,0],data[:,1],c = 'b', marker='o', facecolor="w", edgecolor='b'))
        
        #plot neighborhood grid
        h1 = []; h2 = []; handles.append(h1); handles.append(h2)
        for g in range(len(neighbor)):
            h1.append(plb.plot(centers[neighbor[g,:],0],centers[neighbor[g,:],1],'k')[0])
        for g in range(len(neighbor)):
            h2.append(plb.plot(centers[neighbor[:,g],0],centers[neighbor[:,g],1],'k')[0])

    else:
        # the graphs already exist, we have to plot the data
        # update centers
        handles[1].set_offsets(centers)
        # update paths
        for g in range(len(neighbor)):
            handles[3][g].set_xdata(centers[neighbor[g,:],0])
            handles[3][g].set_ydata(centers[neighbor[g,:],1])
            handles[4][g].set_xdata(centers[neighbor[:,g],0])
            handles[4][g].set_ydata(centers[neighbor[:,g],1])

    
    #take care of the zoom
    xmax = np.max(data[:,0] + 1)
    xmin = np.min(data[:,0] - 1)
    ymax = np.max(data[:,1] + 1)
    ymin = np.min(data[:,1] - 1)
    xmax = max(xmax, np.max(centers[:,0] + 0.5))
    xmin = min(xmin, np.min(centers[:,0] - 0.5))
    ymax = max(ymax, np.max(centers[:,1] + 0.5))
    ymin = min(ymin, np.min(centers[:,1] - 0.5))

    plb.axis(xmin = xmin, xmax = xmax, ymin=ymin, ymax=ymax)
    return handles
    

def som_step(centers,data,neighbor,eta,sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension (in this case 2)
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """
    
    size_k = int(np.sqrt(len(centers)))
    
    #find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))
    #update winner
    centers[b,:] += eta * (data - centers[b,:])

    a,b = np.nonzero(neighbor == b)
    for j in range(size_k**2):
        a1,b1 = np.nonzero(neighbor==j)
        if j != b:
    #compute the discount to the update via the neighborhood function        
            disc=gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
        else:
            disc=0
    #update non winners according to the neighborhood function    
        centers[j,:] += disc * eta * (data - centers[j,:])

def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2)) / np.sqrt(2*np.pi) /p[1]


if __name__ == "__main__":
    kohonen()
