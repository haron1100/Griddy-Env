import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from griddy_render import *
from gym.envs.classic_control import rendering

class GriddyEnv(gym.Env):
    """
    Description:
        A grid world where you have to reach the goal
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 4)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Reward:
        Reward is 0 for every step taken and 1 when goal is reached
    Starting State:
        Agent starts in random position and goal is always bottom right
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, width=4, height=4):
        self.n_squares_height = width
        self.n_squares_width = height

        self.OBJECT_TO_IDX = {
            'goal':1,
            'wall':2,
            'agent':3
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width))

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, random_goal=False):
        self.n_steps=0
        state = np.full((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width), 0)
        if random_goal:
            agent_pos, goal_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width), 2, replace=False)
            agent_pos, goal_pos = (agent_pos//self.n_squares_width, agent_pos%self.n_squares_width), (goal_pos//self.n_squares_width, goal_pos%self.n_squares_width)
            state[0, goal_pos[0], goal_pos[1]] = 1
        else:
            agent_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width-1), 1, replace=False)[0]
            agent_pos = (agent_pos//self.n_squares_width, agent_pos%self.n_squares_width)
            state[0, self.n_squares_height-1, self.n_squares_width-1] = 1
            
        state[2, agent_pos[0], agent_pos[1]] = 1
        self.state = state
        self.steps_beyond_done = None
        return np.copy(self.state )

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.n_steps+=1
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[1]-=1
        elif action==1:
            new_agent_pos[1]+=1
        elif action==2:
            new_agent_pos[0]-=1
        elif action==3:
            new_agent_pos[0]+=1
        new_agent_pos[0] = np.clip(new_agent_pos[0], 0, self.n_squares_height-1)
        new_agent_pos[1] = np.clip(new_agent_pos[1], 0, self.n_squares_width-1)
        
        self.state[2, agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[2, new_agent_pos[0], new_agent_pos[1]] = 1 #moved to this position
        
        #check if done
        done=False
        if np.all(np.array(goal_pos)==new_agent_pos):
            done=True
            
        #assign reward
        if not done:
            reward = 0
            if self.n_steps>=1000:
                self.steps_beyond_done = 0
                done=True
        elif self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
            reward = 1
        else:
            if self.steps_beyond_done >= 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.copy(self.state), reward, done, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add invisible squares for visualising state values
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.squares = [[rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            sq_transforms = [[rendering.Transform() for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            for i in range(0, self.n_squares_height):
                for j in range(0, self.n_squares_width):
                    self.squares[i][j].add_attr(sq_transforms[i][j])
                    self.viewer.add_geom(self.squares[i][j])
                    sq_x, sq_y = self.convert_pos_to_xy((i, j), (square_size_width, square_size_height))
                    sq_transforms[i][j].set_translation(sq_x, sq_y)
                    self.squares[i][j].set_color(1, 1, 1)
            #horizontal grid lines
            for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #the agent
            #self.agent = rendering.Image('robo.jpg', width=square_size_width/2, height=square_size_height/2)
            l, r, t, b = -square_size_width/4, square_size_width/4, square_size_height/4, -square_size_height/4
            self.agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/4, height=square_size_height/4)
            self.goal.set_color(1,0,1)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]

        agent_x, agent_y = self.convert_pos_to_xy(agent_pos, (square_size_width, square_size_height))
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_x, goal_y = self.convert_pos_to_xy(goal_pos, (square_size_width, square_size_height))
        self.goaltrans.set_translation(goal_x, goal_y)
        if values is not None:
            maxval, minval = values.max(), values.min()
            rng = maxval-minval
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    if rng==0: col=1
                    else: col=(maxval-val)/rng
                    self.squares[i][j].set_color(col, 1, col)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1]+0.5) * size[0]
        y = (self.n_squares_height-pos[0]-0.5) * size[1]
        return x, y

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

'''values = np.array([[0.73509189, 0.77378094, 0.81450625, 0.857375  ],
       [0.77378094, 0.81450625, 0.857375  , 0.9025    ],
       [0.81450625, 0.857375  , 0.9025    , 0.95      ],
       [0.857375  , 0.9025    , 0.95      , 0        ]])
values = np.array([[0, 0, 0, 0  ],
       [0, 0, 0  , 0    ],
       [0, 0  , 0    , 0      ],
       [0, 0    , 0      , 0        ]])
env=GriddyEnv()
env.reset()
env.render(values)'''

class GriddyEnvAnton(gym.Env):
    """
    Description:
        A grid world where you have to reach the goal
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 6)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Kinds of entities:
        Agent - small black square
        goal - small purple circle
        square - a white square in the grid
        gold coin - a small yellow circle, to only exist until picked up by agent
        thorns/flame - red triangle
        void/gap - a blue square the size of a grid square
        arrow - looks like arrow pushes agent off square to neighbouring square, in direction of arrow
    
    Reward:
        Reward is 0 when stepping into empty square, +10 when stepping into goal, +1 when picking up gold coin, 
        -1 when stepping on thorns, and -100 when stepping in hole/gap.
    Starting State:
        State can be set by setting up position of entities using dictionary, or randomly initialise positions
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
#TODO: add logic preventing infinite loops from arrows pointing towards each other

    def __init__(self, width=4, height=4, object_coordinates = None):
        self.n_squares_height = width
        self.n_squares_width = height

        self.OBJECT_TO_IDX = { # objects will be coded by a 1 in the positions they occupy, and a 0 if absent, except for the arrows
            'goal':0,
            'void':1,
            'agent':2,
            'coin':3,
            'thorns':4,
            'arrow':5 # arrows will be represented by 0 if absent, 
                        # and one of [1,2,3,4] representing the direction in which to push the agent
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width))

        self.seed()
        self.viewer = None
        self.state = None
        self.has_been_reset_before = False
        self.moved_into_wall = False
        self.done = False

        self.n_steps = None
        self.steps_beyond_done = None
        
        if object_coordinates is not None:
            self.setup(object_coordinates)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        return np.copy(self.state )
    
    def setup(self, object_coordinates): 
        self.n_steps = 0
    #object_coordinates is a dictionary with the positions of each object, 
    # e.g. {'goal':[(3,3)],  'void':[(2,2),(0,1)], 'agent':[(2,3)], 'coin':[(1,2),(3,0)], 'thorns':[(3,2), (0,0)], 'arrow': [(3,1,1)]}
    # or {'goal':[(3,0)], 'agent':[(0,0)], 'arrow': [(1,0,4),(2,0,2),(2,1,3),(1,1,3),(0,1,2),(1,2,1),(2,2,4),(2,3,1)]}
        object_keys = ['goal', 'void', 'agent', 'coin', 'thorns', 'arrow']
        state = np.full((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width), 0)

        for key in object_keys:
            if key in object_coordinates:
                coords = object_coordinates[key]
                for coord in coords:
                    if key == 'arrow':
                        state[self.OBJECT_TO_IDX[key]][coord[0],coord[1]] = coord[2]
                    else:
                        state[self.OBJECT_TO_IDX[key]][coord[0],coord[1]] = 1

        self.state = state
        self.has_been_reset_before = True # so that reset doesn't automatically randomise the object positions
        self.done = False
    
    def reset(self, randomise=False):
        #if randomise we have a random new starting position with exactly one of each object
        if randomise or not self.has_been_reset_before:
            self.close()
            self.has_been_reset_before = True
            state = np.full((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width), 0)
            object_keys = ['goal', 'void', 'agent', 'coin', 'thorns', 'arrow']
            object_locations = \
                np.random.choice(range(self.n_squares_height*self.n_squares_width), len(self.OBJECT_TO_IDX), replace=False)
            object_inds = {object_keys[i]:(object_locations[i]//self.n_squares_width, object_locations[i]%self.n_squares_width) \
                           for i in range(len(self.OBJECT_TO_IDX))}
            for key in object_keys:
                if key == 'arrow':
                    state[self.OBJECT_TO_IDX[key]][object_inds[key]] = np.random.choice([1,2,3,4])
                else:
                    state[self.OBJECT_TO_IDX[key]][object_inds[key]] = 1
            
        #else the objects keep their original position, except for the agent which gets a random new starting position
        else:
            state = np.copy(self.state)
            state[self.OBJECT_TO_IDX['agent']] = np.zeros((self.n_squares_height, self.n_squares_width))
            object_keys = ['goal', 'void', 'coin', 'thorns', 'arrow']
            excluded_inds = set() #set of indices to be excluded from new agent position
            for key in object_keys:
                object_location = np.nonzero(self.state[self.OBJECT_TO_IDX[key]])
                for i in range(len(object_location[0])):
                    excluded_inds.add(object_location[0][i]*self.n_squares_width + object_location[1][i])
            agent_indices = [i for i in range(self.n_squares_height*self.n_squares_width) if i not in excluded_inds]
            agent_pos = np.random.choice(agent_indices, 1, replace=False)[0]
            agent_pos = (agent_pos//self.n_squares_width, agent_pos%self.n_squares_width)
            state[self.OBJECT_TO_IDX['agent']][agent_pos] = 1
            #resetting picked up coins
            state[self.OBJECT_TO_IDX['coin']] = np.abs(state[self.OBJECT_TO_IDX['coin']])
            
        self.state = state
        self.n_steps = 0
        self.steps_beyond_done = None
        self.done = False
        return np.copy(self.state )

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        goal_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['goal']] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['agent']] == 1)))[0]
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[1]-=1
        elif action==1:
            new_agent_pos[1]+=1
        elif action==2:
            new_agent_pos[0]-=1
        elif action==3:
            new_agent_pos[0]+=1
        if (new_agent_pos[0] < 0 or new_agent_pos[0] > self.n_squares_height-1) \
            or (new_agent_pos[1] < 0 or new_agent_pos[1] > self.n_squares_width-1):
            self.moved_into_wall = True
        new_agent_pos[0] = np.clip(new_agent_pos[0], 0, self.n_squares_height-1)
        new_agent_pos[1] = np.clip(new_agent_pos[1], 0, self.n_squares_width-1)
        
        self.state[self.OBJECT_TO_IDX['agent'], agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[self.OBJECT_TO_IDX['agent'], new_agent_pos[0], new_agent_pos[1]] = 1 #moved to this position
        
        #check if done
        if np.all(np.array(goal_pos)==new_agent_pos):
            self.done=True
            
        #assign reward
        reward = 0
        
        #assuming here there are no overlapping objects
        if self.state[self.OBJECT_TO_IDX['goal'], new_agent_pos[0], new_agent_pos[1]] == 1: reward +=10
        if self.state[self.OBJECT_TO_IDX['void'], new_agent_pos[0], new_agent_pos[1]] == 1: reward -=100
        if self.state[self.OBJECT_TO_IDX['thorns'], new_agent_pos[0], new_agent_pos[1]] == 1: reward -=1
        if self.state[self.OBJECT_TO_IDX['coin'], new_agent_pos[0], new_agent_pos[1]] == 1:
            reward +=1
            #coin has been picked up, change value to -1 to get rid of reward, but remember position for re-rendering
            self.state[self.OBJECT_TO_IDX['coin'], new_agent_pos[0], new_agent_pos[1]] = -1 
        if self.state[self.OBJECT_TO_IDX['arrow'], new_agent_pos[0], new_agent_pos[1]] != 0 and not self.moved_into_wall:
        # the logic being that if the agent moved into a wall, it is because the arrow made it, and we want to avoid infinite loops
        #TODO: add logic preventing infinite loops from arrows pointing towards each other
            return self.step(self.state[self.OBJECT_TO_IDX['arrow'], new_agent_pos[0], new_agent_pos[1]] - 1)
            
        
        self.moved_into_wall = False #resetting the check
        self.n_steps+=1 #moved here so that arrow move doesn't count as a step
            
        if not self.done:
            if self.n_steps>=1000:
                self.steps_beyond_done = 0
                self.done=True        
        elif self.done and self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
        elif self.done and self.steps_beyond_done >= 0:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            reward = 0
            self.steps_beyond_done += 1

        return np.copy(self.state), reward, self.done, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add invisible squares for visualising state values
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.squares = [[rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            sq_transforms = [[rendering.Transform() for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            for i in range(0, self.n_squares_height):
                for j in range(0, self.n_squares_width):
                    self.squares[i][j].add_attr(sq_transforms[i][j])
                    self.viewer.add_geom(self.squares[i][j])
                    sq_x, sq_y = self.convert_pos_to_xy((i, j), (square_size_width, square_size_height))
                    sq_transforms[i][j].set_translation(sq_x, sq_y)
                    self.squares[i][j].set_color(1, 1, 1)
            #horizontal grid lines
            for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #the agent
            #self.agent = rendering.Image('robo.jpg', width=square_size_width/2, height=square_size_height/2)
            l, r, t, b = -square_size_width/4, square_size_width/4, square_size_height/4, -square_size_height/4
            self.agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the voids
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.voids = [ rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for i in \
                          range(np.sum(self.state[self.OBJECT_TO_IDX['void']])) ]
            self.voidtrans = [rendering.Transform() for i in range(len(self.voids))]
            for i in range(len(self.voids)):
                self.voids[i].set_color(0,0,1)
                self.voids[i].add_attr(self.voidtrans[i])
                self.viewer.add_geom(self.voids[i])
            #the goal
            self.goal = make_oval(width=square_size_width/4, height=square_size_height/4)
            self.goal.set_color(1,0,1)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
            #the coins
            self.coins = [ make_oval(width=square_size_width/4, height=square_size_height/4) for i in \
                          range(np.sum(self.state[self.OBJECT_TO_IDX['coin']])) ]
            self.cointrans = [rendering.Transform() for i in range(len(self.coins))]
            for i in range(len(self.coins)):
                self.coins[i].set_color(1,1,0)
                self.coins[i].add_attr(self.cointrans[i])
                self.viewer.add_geom(self.coins[i])
            #the thorns
            l, r, t, b = -square_size_width/4, square_size_width/4, square_size_height/3, -square_size_height/3
            self.thorns = [ rendering.FilledPolygon([(l,b), (0,t), (r,b)]) for i in \
                          range(np.sum(self.state[self.OBJECT_TO_IDX['thorns']])) ]
            self.thornstrans = [rendering.Transform() for i in range(len(self.thorns))]
            for i in range(len(self.thorns)):
                self.thorns[i].set_color(1,0,0)
                self.thorns[i].add_attr(self.thornstrans[i])
                self.viewer.add_geom(self.thorns[i])
            #the leftarrows
            x1, x2, x3 = -0.35*square_size_width, -0.1*square_size_width, 0.3*square_size_width
            y1, y2, y3, y4, y5 = -0.14*square_size_height, -0.03*square_size_height, 0,\
                0.03*square_size_height, 0.14*square_size_height
            self.leftarrows = [rendering.FilledPolygon([(x1,y3),(x2,y5),(x2,y4),(x3,y4),(x3,y2),(x2,y2),(x2,y1)]) for i in \
                              range(np.sum(self.state[self.OBJECT_TO_IDX['arrow']]==1))]
            self.leftarrowtrans = [rendering.Transform() for i in range(len(self.leftarrows))]
            for i in range(len(self.leftarrows)):
                self.leftarrows[i].set_color(0,0,0)
                self.leftarrows[i].add_attr(self.leftarrowtrans[i])
                self.viewer.add_geom(self.leftarrows[i])            
            #the rightarrows
            x1, x2, x3 = -0.3*square_size_width, 0.1*square_size_width, 0.35*square_size_width
            y1, y2, y3, y4, y5 = -0.14*square_size_height, -0.03*square_size_height, 0,\
                0.03*square_size_height, 0.14*square_size_height
            self.rightarrows = [rendering.FilledPolygon([(x2,y2),(x1,y2),(x1,y4),(x2,y4),(x2,y5),(x3,y3),(x2,y1)]) for i in \
                              range(np.sum(self.state[self.OBJECT_TO_IDX['arrow']]==2))]
            self.rightarrowtrans = [rendering.Transform() for i in range(len(self.rightarrows))]
            for i in range(len(self.rightarrows)):
                self.rightarrows[i].set_color(0,0,0)
                self.rightarrows[i].add_attr(self.rightarrowtrans[i])
                self.viewer.add_geom(self.rightarrows[i])             
            #the uparrows
            x1, x2, x3, x4, x5 = -0.14*square_size_height, -0.03*square_size_height, 0,\
                0.03*square_size_height, 0.14*square_size_height
            y1, y2, y3 = -0.3*square_size_width, 0.1*square_size_width, 0.35*square_size_width
            self.uparrows = [rendering.FilledPolygon([(x2,y2),(x1,y2),(x3,y3),(x5,y2),(x4,y2),(x4,y1),(x2,y1)]) for i in \
                              range(np.sum(self.state[self.OBJECT_TO_IDX['arrow']]==3))]
            self.uparrowtrans = [rendering.Transform() for i in range(len(self.uparrows))]
            for i in range(len(self.uparrows)):
                self.uparrows[i].set_color(0,0,0)
                self.uparrows[i].add_attr(self.uparrowtrans[i])
                self.viewer.add_geom(self.uparrows[i])
            #the downarrows
            x1, x2, x3, x4, x5 = -0.14*square_size_height, -0.03*square_size_height, 0,\
                0.03*square_size_height, 0.14*square_size_height
            y1, y2, y3 = -0.35*square_size_width, -0.1*square_size_width, 0.3*square_size_width
            self.downarrows = [rendering.FilledPolygon([(x3,y1),(x1,y2),(x2,y2),(x2,y3),(x4,y3),(x4,y2),(x5,y2)]) for i in \
                              range(np.sum(self.state[self.OBJECT_TO_IDX['arrow']]==4))]
            self.downarrowtrans = [rendering.Transform() for i in range(len(self.downarrows))]
            for i in range(len(self.downarrows)):
                self.downarrows[i].set_color(0,0,0)
                self.downarrows[i].add_attr(self.downarrowtrans[i])
                self.viewer.add_geom(self.downarrows[i])
                
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['goal']] == 1)))[0]
        void_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['void']] == 1)))
        agent_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['agent']] == 1)))[0]
        coin_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['coin']] != 0)))
        thorns_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['thorns']] == 1)))
        leftarrow_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['arrow']] == 1)))
        rightarrow_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['arrow']] == 2)))
        uparrow_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['arrow']] == 3)))
        downarrow_pos = list(zip(*np.where(self.state[self.OBJECT_TO_IDX['arrow']] == 4)))

        
        goal_x, goal_y = self.convert_pos_to_xy(goal_pos, (square_size_width, square_size_height))
        self.goaltrans.set_translation(goal_x, goal_y)
        
        for i in range(len(self.voids)):
            void_x, void_y = self.convert_pos_to_xy(void_pos[i], (square_size_width, square_size_height))
            self.voidtrans[i].set_translation(void_x, void_y)

        agent_x, agent_y = self.convert_pos_to_xy(agent_pos, (square_size_width, square_size_height))
        self.agenttrans.set_translation(agent_x, agent_y)
        
        for i in range(len(self.coins)):
            if coin_pos[i] == agent_pos: #making sure there are no white disks on top of the agent
                self.coins[i].set_color(0,0,0)
            elif self.state[self.OBJECT_TO_IDX['coin']][coin_pos[i]] == -1:
                self.coins[i].set_color(1,1,1)
            else:
                self.coins[i].set_color(1,1,0)
            coin_x, coin_y = self.convert_pos_to_xy(coin_pos[i], (square_size_width, square_size_height))
            self.cointrans[i].set_translation(coin_x, coin_y)
        
        for i in range(len(self.thorns)):
            thorns_x, thorns_y = self.convert_pos_to_xy(thorns_pos[i], (square_size_width, square_size_height))
            self.thornstrans[i].set_translation(thorns_x, thorns_y)

        for i in range(len(self.leftarrows)):
            leftarrow_x, leftarrow_y = self.convert_pos_to_xy(leftarrow_pos[i], (square_size_width, square_size_height))
            self.leftarrowtrans[i].set_translation(leftarrow_x, leftarrow_y)
        
        for i in range(len(self.rightarrows)):
            rightarrow_x, rightarrow_y = self.convert_pos_to_xy(rightarrow_pos[i], (square_size_width, square_size_height))
            self.rightarrowtrans[i].set_translation(rightarrow_x, rightarrow_y)

        for i in range(len(self.uparrows)):
            uparrow_x, uparrow_y = self.convert_pos_to_xy(uparrow_pos[i], (square_size_width, square_size_height))
            self.uparrowtrans[i].set_translation(uparrow_x, uparrow_y)
        
        for i in range(len(self.downarrows)):
            downarrow_x, downarrow_y = self.convert_pos_to_xy(downarrow_pos[i], (square_size_width, square_size_height))
            self.downarrowtrans[i].set_translation(downarrow_x, downarrow_y)
            
        if values is not None:
            maxval, minval = values.max(), values.min()
            rng = maxval-minval
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    if rng==0: col=1
                    else: col=(maxval-val)/rng
                    self.squares[i][j].set_color(col, 1, col)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1]+0.5) * size[0]
        y = (self.n_squares_height-pos[0]-0.5) * size[1]
        return x, y

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

