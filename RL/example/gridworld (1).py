from pylab import *                                                                                                  
import numpy
from time import sleep

class Gridworld:
    """
    A class that implements a quadratic NxN gridworld. 
    
    Methods:
    
    learn(N_trials=100)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number. 
    navigation_map()     : Plot the movement direction with the highest Q-value for all positions.
    """    
        
    def __init__(self,N,reward_position=(0,0),obstacle=False, lambda_eligibility=0.):
        """
        Creates a quadratic NxN gridworld. 

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        reward_position = (x_coordinate,y_coordinate): the reward location
        obstacle = True:  Add a wall to the gridworld.
        """    
        
        # gridworld size
        self.N = N

        # reward location
        self.reward_position = reward_position        

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 1.
        self.reward_at_wall   = -0.5

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = 0.5
                                                                                                  
        # learning rate
        self.eta = 0.1

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.99

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility
    
        # is there an obstacle in the room?
        self.obstacle = obstacle

        # initialize the Q-values etc.
        self._init_run()

    def run(self,N_trials=10,N_runs=1):
        self.latencies = zeros(N_trials)
        
        for run in range(N_runs):
            self._init_run()
            latencies = self._learn_run(N_trials=N_trials)
            self.latencies += latencies/N_runs

    def visualize_trial(self):
        """
        Run a single trial with a graphical display that shows in
                red   - the position of the agent
                blue  - walls/obstacles
                green - the reward position

        Note that for the simulation, exploration is reduced -> self.epsilon=0.1
    
        """
        # store the old exploration/exploitation parameter
        epsilon = self.epsilon

        # favor exploitation, i.e. use the action with the
        # highest Q-value most of the time
        self.epsilon = 0.1

        self._run_trial(visualize=True)

        # restore the old exploration/exploitation factor
        self.epsilon = epsilon

    def learning_curve(self,log=False,filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        figure()
        xlabel('trials')
        ylabel('time to reach target')
        latencies = array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)

        if not log:
            plot(self.latencies)
        else:
            semilogy(self.latencies)

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = numpy.zeros((self.N,self.N))
        self.y_direction = numpy.zeros((self.N,self.N))
        
        self.actions = argmax(self.Q[:,:,:],axis=2)
        self.y_direction[self.actions==0] = 1.
        self.y_direction[self.actions==1] = -1.
        self.y_direction[self.actions==2] = 0.
        self.y_direction[self.actions==3] = 0.
        
        self.x_direction[self.actions==0] = 0.
        self.x_direction[self.actions==1] = 0.
        self.x_direction[self.actions==2] = 1.
        self.x_direction[self.actions==3] = -1.
        
        figure()
        quiver(self.x_direction,self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

    def reset(self):
        """
        Reset the Q-values (and the latency_list).
        
        Instant amnesia -  the agent forgets everything he has learned before    
        """
        self.Q = numpy.random.rand(self.N,self.N,4)
        self.latency_list = []

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position.
        The figure consists of 4 subgraphs, each of which shows the Q-values 
        colorcoded for one of the actions.
        """
        figure()
        for i in range(4):
            subplot(2,2,i+1)
            imshow(self.Q[:,:,i],interpolation='nearest',origin='lower',vmax=1.1)
            if i==0:
                title('Up')
            elif i==1:
                title('Down')
            elif i==2:
                title('Right')
            else:
                title('Left')
    
            colorbar()
        draw()
    
    ###############################################################################################
    # The remainder of methods is for internal use and only relevant to those of you
    # that are interested in the implementation details
    ###############################################################################################
        
    
    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        self.Q = 0.01 * numpy.random.rand(self.N,self.N,4) + 0.1
        self.e = numpy.zeros((self.N,self.N,4))
        
        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None

    def _learn_run(self,N_trials=10):
        """
        Run a learning period consisting of N_trials trials. 
        
        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.
        
        """
        for trial in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial()
            self.latency_list.append(latency)

        return array(self.latency_list)

    def _run_trial(self,visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """
        # choose the initial position and make sure that its not in the wall
        while True:
            self.x_position = numpy.random.randint(self.N)
            self.y_position = numpy.random.randint(self.N)
            if not self._is_wall(self.x_position,self.y_position):
                break
        
        # initialize the latency (time to reach the target) for this trial
        latency = 0.

        # start the visualization, if asked for
        if visualize:
            self._init_visualization()    
            
        # run the trial
        self._choose_action()
        while not self._arrived():
            self._update_state()
            self._choose_action()    
            self._update_Q()
            if visualize:
                self._visualize_current_state()
        
            latency = latency + 1

        if visualize:
            self._close_visualization()
        return latency

    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.e
        self.e[self.x_position_old, self.y_position_old,self.action_old] += 1.
        
        # update the Q-values
        if self.action_old != None:
            self.Q +=     \
                self.eta * self.e *\
                (self._reward()  \
                - ( self.Q[self.x_position_old,self.y_position_old,self.action_old] \
                - self.gamma * self.Q[self.x_position, self.y_position, self.action] )  )

    def _choose_action(self):    
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action 
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(4)
        else:
            self.action = argmax(self.Q[self.x_position,self.y_position,:])    
    
    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        return (self.x_position == self.reward_position[0] and self.y_position == self.reward_position[1])

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the 
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.    
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position
        
        # update the agents position according to the action
        # move to the down?
        if self.action == 0:
            self.x_position += 1
        # move to the up
        elif self.action == 1:
            self.x_position -= 1
        # move right?
        elif self.action == 2:
            self.y_position += 1
        # move left?
        elif self.action == 3:
            self.y_position -= 1
        else:
            print "There must be a bug. This is not a valid action!"
                        
        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
        else:
            self._wall_touch = False

    def _is_wall(self,x_position=None,y_position=None):    
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have 
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position < 0 or x_position >= self.N or y_position < 0 or y_position >= self.N:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            if y_position == self.N/2 and x_position>self.N/2:
                return True

        # if none of the above is the case, this position is not a wall
        return False 
            
    def _visualize_current_state(self):
        """
        Show the gridworld. The squares are colored in 
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """

        # set the agents color
        self._display[self.x_position_old,self.y_position_old,0] = 0
        self._display[self.x_position_old,self.y_position_old,1] = 0
        self._display[self.x_position,self.y_position,0] = 1
        if self._wall_touch:
            self._display[self.x_position,self.y_position,1] = 1
            
        # set the reward locations
        self._display[self.reward_position[0],self.reward_position[1],1] = 1

        # update the figure
        self._visualization.set_data(self._display)
        #close()
        #imshow(self._display,interpolation='nearest',origin='lower')
        #show()
        
        draw()
        
        # and wait a little while to control the speed of the presentation
        sleep(0.2)
        
    def _init_visualization(self):
        
        # create the figure
        figure()
        # initialize the content of the figure (RGB at each position)
        self._display = numpy.zeros((self.N,self.N,3))

        # position of the agent
        self._display[self.x_position,self.y_position,0] = 1
        self._display[self.reward_position[0],self.reward_position[1],1] = 1
        
        for x in range(self.N):
            for y in range(self.N):
                if self._is_wall(x_position=x,y_position=y):
                    self._display[x,y,2] = 1.

                self._visualization = imshow(self._display,interpolation='nearest',origin='lower')
        
    def _close_visualization(self):
        print "Press <return> to proceed..."
        raw_input()
        close()
