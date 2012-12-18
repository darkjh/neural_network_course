from pylab import *
from numpy import *
from time import sleep

class RL:
    """
    A class that implements a continuous gridworld.

    Methods:

    learn(N_trials=100)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number.
    navigation_map()     : Plot the movement direction with the highest Q-value for all positions.
    """

    def __init__(self, N=20, lambda_eligibility=0.95):
        """
        Creates a quadratic NxN gridworld.

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        obstacle = True:  Add a wall to the gridworld.
        """

        # input neuron size
        self.N = N

        # width of the 2-D state space
        self.width = 1.0

        # step length
        self.step_length = 0.03

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 10.0
        self.reward_at_wall   = -2.0

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = 0.5

        # learning rate
        self.eta = 0.005

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility

        # sigma param for gaussian basis function
        self.sigma = 0.05

        # max iteration
        self.iter_max = 10000

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
        for run in range(20):
            l = self._run_trial(visualize=True)
            print "step: " + str(l)

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
        self.Q = numpy.random.rand(self.N,self.N,8)
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

    def _to_position(self, pos):
        """
        Convert the index of input neuron into absolute coordinate on state space
        """
        return pos / 19

    def _basis_function(self, j_x, j_y, x_pos=None, y_pos=None):
        """
        Gaussian basis function, used to code the 2-D state space
        """
        if x_pos == None and y_pos == None:
            x = self.x_pos
            y = self.y_pos
        else:
            x = x_pos
            y = y_pos

        return exp(-((j_x - x)**2 + (j_y - y)**2) / (2 * self.sigma**2))


    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        self.Q = zeros(8)
        self.Q_old = None
        self.w = zeros((self.N, self.N, 8)) 
#	self.w = 0.01 * random.rand(self.N, self.N, 8) + 0.1
        self.e = zeros((self.N, self.N, 8))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_pos = None
        self.y_pos = None
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
#	    self.epsilon = self.epsilon * self.epsilon    
#	    print "epsilon = " + str(self.epsilon)
            self.latency_list.append(latency)

        return array(self.latency_list)

    def _run_trial(self,visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """

        # initial position of the agent
        self.x_pos = 0.1
        self.y_pos = 0.1

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
            self._update_W()
            if visualize:
                self._visualize_current_state()

            latency = latency + 1
            if latency % 500 == 0:
                print str(latency)
            # if latency >= self.iter_max:
              #  break

	
	print "latency = "+str(latency)
        if visualize:
            self._close_visualization()

        return latency

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_pos_old = self.x_pos
        self.y_pos_old = self.y_pos

        # update the agents position according to the action
        if self.action in range(8):
            self.x_pos = self.x_pos_old + self.step_length * cos(2*pi*self.action/8)
            self.y_pos = self.y_pos_old + self.step_length * sin(2*pi*self.action/8)
        else:
            print "There must be a bug. This is not a valid action!"

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_pos = self.x_pos_old
            self.y_pos = self.y_pos_old
            self._wall_touch = True
        else:
            self._wall_touch = False

    def _update_W(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        #self.e = self.gamma * self.lambda_eligibility * self.e
        self.e = self.lambda_eligibility * self.e
        for i in range(self.N):
            for j in range(self.N):
                self.e[i, j, self.action_old] += self._basis_function(self._to_position(i), self._to_position(j), self.x_pos_old, self.y_pos_old)
	        # update the W_j,a
	        if self.action_old != None and self.Q_old != None:
        	    TD = self._reward() + self.gamma*self.Q[self.action] - self.Q_old[self.action_old]
	            self.w[i, j, self.action_old] += self.eta * TD * self.e[i, j, self.action_old]
#		    if TD != 0:
#			print "weight = "+str(self.w[i, j, self.action_old])
		else :
		    print "error condition"

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        # calculate Q value for each action
        self.Q_old = self.Q
        self.Q = zeros(8)
        for a in range(8):
            sum = 0.0
            for x in range(self.N):
                for y in range(self.N):
                    sum += self._basis_function(self._to_position(x), self._to_position(y)) * self.w[x, y, a]
            self.Q[a] = sum

        if random.rand() < self.epsilon:
            self.action = random.randint(8)
        else:
            self.action = argmax(self.Q[:])

    def _arrived(self):
        """
        Check if the agent has arrived in the goal circle.
        """
        return (power(self.x_pos - 0.8, 2) + power(self.y_pos - 0.8, 2) < power(0.1, 2))

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
    
    def _is_wall(self,x_position=None,y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_pos
            y_position = self.y_pos

        # check of the agent is trying to leave the gridworld
        if x_position < 0 or x_position > self.width or y_position < 0 or y_position > self.width:
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
        self.plot.set_xdata(self.x_pos)
        self.plot.set_ydata(self.y_pos)

        draw()

        # and wait a little while to control the speed of the presentation
        sleep(0.01)

    def _init_visualization(self):

        # turn on interactive mode
        ion()
        axis([0.0, 1.0, 0.0, 1.0])
        gca().set_aspect('equal')

        # draw the goal area
        goal = Circle((.8, .8), radius=.1, fill=False)
        gca().add_patch(goal)

        line, = plot(0.1, 0.1, 'bo')
        self.plot = line

    def _close_visualization(self):
        print "Press <return> to proceed..."
        raw_input()
        close()
