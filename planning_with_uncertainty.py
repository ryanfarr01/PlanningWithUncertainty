#!/usr/bin/env python

'''
Author: Ryan Farr
Date: 11/15/2016
File: planning_with_uncertainty.py - file contains all structures and algorithms to run and display
    the results of breadth-first-search (both determininstic and stochastic), value iteration,
    and policy iteration. Reads in .txt map files, which should always be the first parameter when
    using the CLI or API

Available action sets are:
    'actions_1' - this allows for up, down, left, and right
    'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right

Available Probability sets are:
    'prob_1' - 80% chance of going to target, 10% chance of going perpendicular in each direction
    'prob_2' - 70% chance of going to target, 10% chance of going perpendicular, 5% chance of going between
    'prob_3' - 60% chance of going to target, 20% chance of going perpendicular in each direction
    'prob_4' - 80% chance of going to target, 10% chance of going in between it and perpendiculars

Available algorithms are:
    'bfs'                 - Breadth-First Search
    'value_iteration'     - Value Iteration
    'policy_iteration'    - Policy Iteration

Available action strings are:
    'u' - Initial map filled with the up action
    'l' - Initial map filled with the left action
    'r' - Initial map filled with the right action
    'd' - Initial map filled with the down action
    'ne' - Initial map filled with the up-right action
    'nw' - Initial map filled with the up-left action
    'se' - Initial map filled with the down-right action
    'sw' - Initial map filled with the down-left action
    'random' - Initial map filled with the random actions chosen from the chosen action set

Examples using command-line:
python planning_with_uncertainty.py Tests/map1.txt actions_2 prob_1 bfs
python planning_with_uncertainty.py Tests/map0.txt actions_1 prob_2 value_iteration 0.8 0.0 10.0
python planning_with_uncertainty.py Tests/map2.txt actions_2 prob_1 policy_iteration 0.8 0.0 10.0 random -10.0

Examples using API:
    planning_with_uncertainty.run_bfs('Tests/map1.txt', 'actions_2', 'prob_1')
    planning_with_uncertainty.run_value_iteration('Tests/map0.txt', 'actions_1', 'prob_2', 0.8, 0.0, 10.0, False)
    planning_with_uncertainty.run_policy_iteration('Tests/map1.txt', 'actions_2', 'prob_1', 0.8, 0.0, 10.0, random, -10.0, True)
'''
import sys
import numpy as np
import heapq
import matplotlib.pyplot as plotter
import math
from sets  import Set

_DEBUG         = False
_DEBUG_END     = True
_ACTIONS       = ['u','d','l','r']
_ACTIONS_2     = ['u','d','l','r','ne','nw','sw','se']
_PERF_PROBS    = [1.0, 0.0, 0.0] 
_PROBS         = [0.8, 0.0, 0.1]
_PROBS_2       = [0.7, 0.1, 0.05]
_PROBS_3       = [0.6, 0.0, 0.2]
_PROBS_4       = [0.8, 0.1, 0.0]
_X = 1
_Y = 0
_GOAL_COLOR    = 0.75
_INIT_COLOR    = 0.25
_BLACK         = 0.0
_VISITED_COLOR = 0.9
_PATH_COLOR_RANGE = _GOAL_COLOR - _INIT_COLOR


class GridMap:
    '''
    Class to hold a grid map for navigation. Reads in a map.txt file of the format
    0 - free cell, x - occupied cell, g - goal location, i - initial location.
    Additionally provides a simple transition model for grid maps and a convience function
    for displaying maps.
    '''
    def __init__(self, map_path=None):
        '''
        Constructor. Makes the necessary class variables. Optionally reads in a provided map
        file given by map_path.

        map_path (optional) - a string of the path to the file on disk
        '''
        self.rows = None
        self.cols = None
        self.goal = None
        self.init_pos = None
        self.occupancy_grid = None
        if map_path is not None:
            self.read_map(map_path)

    def read_map(self, map_path):
        '''
        Read in a specified map file of the format described in the class doc string.

        map_path - a string of the path to the file on disk
        '''
        map_file = file(map_path,'r')
        lines = [l.rstrip().lower() for l in map_file.readlines()]
        map_file.close()
        self.rows = len(lines)
        self.cols = max([len(l) for l in lines])
        if _DEBUG:
            print 'rows', self.rows
            print 'cols', self.cols
            print lines
        self.occupancy_grid = np.zeros((self.rows, self.cols), dtype=np.bool)
        for r in xrange(self.rows):
            for c in xrange(self.cols):
                if lines[r][c] == 'x':
                    self.occupancy_grid[r][c] = True
                if lines[r][c] == 'g':
                    self.goal = (r,c)
                elif lines[r][c] == 'i':
                    self.init_pos = (r,c)

    def is_goal(self,s):
        '''
        Test if a specifid state is the goal state

        s - tuple describing the state as (row, col) position on the grid.

        Returns - True if s is the goal. False otherwise.
        '''
        return (s[_X] == self.goal[_X] and
                s[_Y] == self.goal[_Y])

    def transition(self, s, a, prob_set, sim):
        '''
        Transition function for the current grid map.

        s - tuple describing the state as (row, col) position on the grid.
        a - the action to be performed from state s
        prob_set - which probability set to use
        sim - whether or not to simulate the action. If true, returns a state. Otherwise,
              returns a list of 2-tuples with state and probability

        returns - s_prime, the state transitioned to by taking action a in state s.
        If the action is not valid (e.g. moves off the grid or into an obstacle)
        returns the current state.
        '''
        # Ensure action stays on the board
        poss = []

        if a == 'u':
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]), s), prob_set[0]))   #up
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]-1), s), prob_set[1])) #up-left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]+1), s), prob_set[1])) #up-right
            poss.append((self.get_grid_point((s[_Y], s[_X]-1), s), prob_set[2]))   #left
            poss.append((self.get_grid_point((s[_Y], s[_X]+1), s), prob_set[2]))   #right
        elif a == 'd':
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]), s), prob_set[0]))   #down
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]-1), s), prob_set[1])) #down-left
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]+1), s), prob_set[1])) #down-right
            poss.append((self.get_grid_point((s[_Y], s[_X]-1), s), prob_set[2]))   #left
            poss.append((self.get_grid_point((s[_Y], s[_X]+1), s), prob_set[2]))   #right
        elif a == 'l':
            poss.append((self.get_grid_point((s[_Y], s[_X]-1), s), prob_set[0]))   #left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]-1), s), prob_set[1])) #up-left
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]-1), s), prob_set[1])) #down-left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]), s), prob_set[2]))   #up
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]), s), prob_set[2]))   #down
        elif a == 'r':
            poss.append((self.get_grid_point((s[_Y], s[_X]+1), s), prob_set[0]))   #right
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]+1), s), prob_set[1])) #up-right
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]+1), s), prob_set[1])) #down-right
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]), s), prob_set[2]))   #down
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]), s), prob_set[2]))   #up
        elif a == 'ne':
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]+1), s), prob_set[0])) #up-right
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]), s), prob_set[1]))   #up
            poss.append((self.get_grid_point((s[_Y], s[_X]+1), s), prob_set[1]))   #left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]-1), s), prob_set[2])) #up-left
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]+1), s), prob_set[2])) #bottom-right
        elif a == 'nw':
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]-1), s), prob_set[0])) #up-left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]), s), prob_set[1]))   #up
            poss.append((self.get_grid_point((s[_Y], s[_X]-1), s), prob_set[1]))   #left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]+1), s), prob_set[2])) #up-right
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]-1), s), prob_set[2])) #bottom-left
        elif a == 'se':
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]+1), s), prob_set[0])) #bottom-right
            poss.append((self.get_grid_point((s[_Y], s[_X]+1), s), prob_set[1]))   #right
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]), s), prob_set[1]))   #bottom
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]+1), s), prob_set[2])) #upper-right
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]-1), s), prob_set[2])) #bottom-left
        elif a == 'sw':
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]-1), s), prob_set[0])) #bottom-left
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]), s), prob_set[1]))   #bottom
            poss.append((self.get_grid_point((s[_Y], s[_X]-1), s), prob_set[1]))   #left
            poss.append((self.get_grid_point((s[_Y]-1, s[_X]-1), s), prob_set[2])) #upper-left
            poss.append((self.get_grid_point((s[_Y]+1, s[_X]+1), s), prob_set[2])) #bottom-right
        else:
            print 'Unknown action:', str(a)

        # if we aren't simulating, just return 'poss'
        if sim != True:
            return poss
        
        # otherwise we need to pick one
        new_pos = list(s[:])
        rand = np.random.random_sample()
        cur_val = 0

        for x in range(0, len(poss)):
            cur_val += poss[x][1] #add the probability of this one
            if rand < cur_val:
                new_pos[_X] = poss[x][0][_X]
                new_pos[_Y] = poss[x][0][_Y]
                break

        return tuple(new_pos)

    def get_grid_point(self, point, original):
        '''
        Function that determines which point point to use - original or calculated

        point - the point being tested against boundaries
        original - the original point, which we default to if point is out of bounds

        returns - either the point or the original. Original if it's out of bounds, is
            the goal, or hits a wall. Point is otherwise returned
        '''
        if point[_X] < 0 or point[_X] >= self.cols or point[_Y] < 0 or point[_Y] >= self.rows or self.is_goal(original) or self.occupancy_grid[point[0], point[1]]:
            return original
        
        return point

    def display_map(self, path=[], visited={}):
        '''
        Visualize the map read in. Optionally display the resulting plan and visisted nodes

        path - a list of tuples describing the path take from init to goal
        visited - a set of tuples describing the states visited during a search
        '''
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)

        # Color all visited nodes if requested
        for v in visited:
            display_grid[v] = _VISITED_COLOR
        # Color path in increasing color from init to goal
        for i, p in enumerate(path):
            disp_col = _INIT_COLOR + _PATH_COLOR_RANGE*(i+1)/len(path)
            display_grid[p] = disp_col

        display_grid[self.init_pos] = _INIT_COLOR

        # Plot display grid for visualization
        imgplot = plotter.imshow(display_grid)
        # Set interpolation to nearest to create sharp boundaries
        imgplot.set_interpolation('nearest')
        # Set color map to diverging style for contrast
        imgplot.set_cmap('spectral')
        plotter.show()

    def display_states(self, map):
        '''
        function that displays a map of the states given in a map

        map - the map of states - should be of the form (value, state)

        returns nothing
        '''
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)
        min_val = 10000000
        max_val = -10000000
        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                if self.is_goal((y, x)):
                    plotter.text(x, y, 'G')
                else:
                    plotter.text(x, y, map[y][x][1])
                if map[y][x][0] > max_val:
                    max_val = map[y][x][0]
                if map[y][x][0] < min_val:
                    min_val = map[y][x][0]

        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                if self.occupancy_grid[y][x]:
                    display_grid[(y, x)] = _BLACK
                else:
                    display_grid[(y,x)] = _INIT_COLOR + ((map[y][x][0] - min_val)/(max_val-min_val))
        display_grid[self.goal] = _GOAL_COLOR

        # Plot display grid for visualization
        imgplot = plotter.imshow(display_grid)
        # Set interpolation to nearest to create sharp boundaries
        imgplot.set_interpolation('nearest')
        # Set color map to diverging style for contrast
        imgplot.set_cmap('spectral')
        plotter.show()

    def display_values(self, map):
        '''
        Function that displays the map of values

        map - grid of the form (value, action)

        returns nothing
        '''
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)

        min_val = 10000000
        max_val = -10000000
        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                plotter.text(x, y, "%.1f" % map[y][x][0], horizontalalignment='center', verticalalignment='center')
                if map[y][x][0] > max_val:
                    max_val = map[y][x][0]
                if map[y][x][0] < min_val:
                    min_val = map[y][x][0]

        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                if self.occupancy_grid[y][x]:
                    display_grid[(y, x)] = _BLACK
                else:
                    display_grid[(y,x)] = _INIT_COLOR + ((map[y][x][0] - min_val)/(max_val-min_val))
        display_grid[self.goal] = _GOAL_COLOR

        # Plot display grid for visualization
        imgplot = plotter.imshow(display_grid)
        # Set interpolation to nearest to create sharp boundaries
        imgplot.set_interpolation('nearest')
        # Set color map to diverging style for contrast
        imgplot.set_cmap('spectral')
        plotter.show()


class SearchNode:
    def __init__(self, s, A, parent=None, parent_action=None, cost=0):
        '''
        s - the state defining the search node
        A - list of actions
        parent - the parent search node
        parent_action - the action taken from parent to get to s
        '''
        self.parent = parent
        self.cost = cost
        self.parent_action = parent_action
        self.state = s[:]
        self.actions = A[:]

    def __str__(self):
        '''
        Return a human readable description of the node
        '''
        return str(self.state) + ' ' + str(self.actions)+' '+str(self.parent)+' '+str(self.parent_action)

def bfs(init_state, f, is_goal, actions):
    '''
    Perform breadth first search on a grid map.

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent

    returns - ((path, action_path), visited) or None if no path can be found
    path - a list of tuples. The first element is the initial state followed by all states
        traversed until the final goal state
    action_path - the actions taken to transition from the initial state to goal state
    '''

    frontier = [] #use as queue
    n0 = SearchNode(init_state, actions)
    visited = set()
    frontier.append(n0)
    while len(frontier) > 0:
        n_i = frontier.pop(0)
        if n_i.state not in visited:
            visited.add(n_i.state)
            if is_goal(n_i.state):
                return(backpath(n_i), visited)
            else:
                for a in actions:
                    s_prime = f(n_i.state, a, _PERF_PROBS, True)
                    n_prime = SearchNode(s_prime, actions, n_i, a)
                    frontier.append(n_prime)
    
    #If we get here, the goal was never reached
    return None

def backpath(node):
    '''
    Function to determine the path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    '''
    path        = []
    action_path = []
    cur_node = node
    
    #Go over each node and itself and the action that its parent took to get there
    while cur_node.parent != None:
        path.insert(0, cur_node.state)
        action_path.insert(0, cur_node.parent_action)
        cur_node = cur_node.parent

    #add in the start node, which should be the cur_node
    path.insert(0, cur_node.state)
    action_path.insert(0, cur_node.parent_action)

    return (path, action_path)

def backpath_stochastic(start, actions, probs, t):
    '''
    Function to determine the path that lead to the specified search node

    start - starting state
    actions - the list of actions taken
    probs - which probability set to use
    t - the transition function

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    '''
    path = []
    path.append(start)

    state = start
    for x in xrange(1, len(actions)):
        a = actions[x]
        state = t(state, a, probs, True)
        path.append(state)

    return (path, actions)

def value_iteration(map, t, discount, action_set, probs, base_reward = 0, goal_reward = 10, corner_reward = 0, use_corners = False):
    '''
    Function that performs value iteration on a map

    map - the map which value iteration is performed upon
    discount - the discount factor (lambda) to be used for value iteration
    action_set - which set of actions to use
    probs - the probability set associated with a given action
    base_reward - the reward for all tiles
    goal_reward - the reward for the goal
    corner_reward - the reward for the corners
    use_corners - bool which tells if we should use different value for corners than base_value

    returns - a grid containing tuples of the form: (value, action) as well as the
        number of iterations required to converge: (grid(value, action), iterations)
    '''
    epsilon = 0.001
    reward_grid = [[base_reward for i in xrange(0, map.cols)] for i in xrange(0, map.rows)]
    if use_corners:
        reward_grid[0][0] = corner_reward
        reward_grid[0][map.cols-1] = corner_reward
        reward_grid[map.rows-1][0] = corner_reward
        reward_grid[map.rows-1][map.cols-1] = corner_reward

    reward_grid[map.goal[_Y]][map.goal[_X]] = goal_reward
    
    # the value grid is the same as the reward grid initially
    value_grid = [None] * map.rows
    for y in xrange(len(value_grid)):
        value_grid[y] = [0] * map.cols
        for x in xrange(len(value_grid[y])):
            r = reward_grid[y][x]
            value_grid[y][x] = (r, r) #g[0] = current, g[1] = previous value

    #Iterate to get value iteration
    needs_iteration = True
    iter = 0
    while(needs_iteration):
        iter += 1
        needs_iteration = False
        for y in xrange(len(value_grid)):
            for x in xrange(len(value_grid[y])):
                if map.is_goal((y, x)):
                    continue
                #get max action value and set as value in grid
                max_val = -sys.maxint - 1 #get minimum value
                for a in action_set:
                    cur_val = 0
                    a_set = t((y, x), a, probs, False)
                    reward = reward_grid[y][x]
                    for s, prob in a_set:
                        s_x = s[_X]
                        s_y = s[_Y]
                        value = value_grid[s_y][s_x][1]
                        cur_val += prob * (reward + (discount * value))
                    
                    if cur_val > max_val:
                        max_val = cur_val
                temp_val = value_grid[y][x][0]
                value_grid[y][x] = (max_val, value_grid[y][x][1])
                
                if abs(value_grid[y][x][0] - temp_val) > epsilon:
                    needs_iteration = True
        update_grid(value_grid)

    #Get the policy
    policy_grid = [None] * map.rows
    for y in xrange(len(policy_grid)):
        policy_grid[y] = [None] * map.cols
        for x in xrange(len(policy_grid[y])):
            val = value_grid[y][x][0]
            max_val = -sys.maxint - 1
            best_action = 'n'
            for a in action_set:
                cur_val = 0
                a_set = t((y, x), a, probs, False)
                for s, prob in a_set:
                    s_x = s[_X]
                    s_y = s[_Y]
                    reward = reward_grid[s_y][s_x]
                    value = value_grid[s_y][s_x][0]
                    cur_val += prob * (reward + (discount * value))
                
                if cur_val > max_val:
                    max_val = cur_val
                    best_action = a
            policy_grid[y][x] = (value_grid[y][x][0], best_action)

    return (policy_grid, iter)

def policy_iteration(map, t, action_grid, discount, action_set, probs, base_reward, goal_reward, corner_reward, use_corners = False):
    '''
    function - performs policy iteration on a given map with initial actions 'action_grid'

    map - the map to be used
    t - the transition function
    action_grid - the current grid
    discount - the discount factor (lambda)
    action_set - the set of actions to be used
    probs - the set of probabilities to be used
    base_reward - the reward for all tiles other than the goal
    goal_reward - the reward to be used for the goal
    corner_reward - cost to be used on corners
    use_corners - whether or not you should use corners

    returns - a map of tuples of the form (value, action) and the number of iterations
        returns: (grid(value, action), iterations)
    '''
    epsilon = 0.001
    reward_grid = [[base_reward for i in xrange(0, map.cols)] for i in xrange(0, map.rows)]
    if use_corners:
        reward_grid[0][0] = corner_reward
        reward_grid[0][map.cols-1] = corner_reward
        reward_grid[map.rows-1][0] = corner_reward
        reward_grid[map.rows-1][map.cols-1] = corner_reward
    reward_grid[map.goal[_Y]][map.goal[_X]] = goal_reward

    # do policy iteration
    iter = 0
    total_iter = 0
    needs_iteration = True
    while needs_iteration:
        # get initial value grid based on actions
        value_grid = [None] * map.rows
        for y in xrange(len(value_grid)):
            value_grid[y] = [0] * map.cols
            for x in xrange(len(value_grid[y])):
                r = reward_grid[y][x]
                value_grid[y][x] = (r, r)

        #Calculate value until converges
        iter += 1
        needs_iteration = False
        needs_iter_value = True
        while needs_iter_value:
            total_iter += 1
            needs_iter_value = False
            for y in xrange(len(value_grid)):
                for x in xrange(len(value_grid[y])):
                    if map.is_goal((y, x)):
                        continue
                    #Compute value
                    cur_val = 0
                    a = action_grid[y][x]
                    a_set = t((y, x), a, probs, False)
                    reward = reward_grid[y][x]
                    for s, prob in a_set:
                        s_x = s[_X]
                        s_y = s[_Y]
                        value = value_grid[s_y][s_x][1]
                        cur_val += prob * (reward + (discount * value))

                    if abs(cur_val - value_grid[y][x][1]) > epsilon:
                        needs_iter_value = True
                    value_grid[y][x] = (cur_val, value_grid[y][x][1])
                
            update_grid(value_grid)

        #change actions
        for y in xrange(len(value_grid)):
            for x in xrange(len(value_grid[y])):
                max_val = -sys.maxint - 1
                best_action = 'u'
                for a in action_set:
                    cur_val = 0
                    a_set = t((y, x), a, probs, False)
                    reward = reward_grid[y][x]
                    for s, prob in a_set:
                        s_x = s[_X]
                        s_y = s[_Y]
                        val = value_grid[s_y][s_x][0]
                        cur_val += prob * (reward + (discount * val))
                    
                    if cur_val > max_val:
                        max_val = cur_val
                        best_action = a
                
                if best_action != action_grid[y][x]:
                    needs_iteration = True
                action_grid[y][x] = best_action

    #Set up return grid
    policy_grid = [None] * map.rows
    for y in xrange(len(value_grid)):
        policy_grid[y] = [None] * map.cols
        for x in xrange(len(value_grid[y])):
            val = value_grid[y][x][0]
            action = action_grid[y][x]
            policy_grid[y][x] = (val, action)

    print('Total number of iterations: ' + str(total_iter))
    return (policy_grid, iter)
    

def update_grid(grid):
    '''
    Function that moves grid values from current to previous

    grid - the grid as a 2D array of tuples that needs updating

    returns - the updated grid
    '''
    for y in xrange(len(grid)):
        for x in xrange(len(grid[y])):
            grid[y][x] = (grid[y][x][0], grid[y][x][0])

    return grid

def make_grid_action(map, a):
    '''
    Function that creates a grid filled with action a
    
    map - the map to be used which stores its dimensions
    a - which action to fill the grid with

    returns - the grid
    '''
    ret = [None] * map.rows
    for y in xrange(len(ret)):
        ret[y] = [None] * map.cols
        for x in xrange(len(ret[y])):
            ret[y][x] = a

    return ret

def make_grid_action_random(map, action_set):
    '''
    Function which creates a grid filled with random actions

    map - the map to be used, which stores its dimensions
    action_set - the set of actions to be chosen from
    
    returns - the grid
    '''
    ret = [None] * map.rows
    for y in xrange(len(ret)):
        ret[y] = [None] * map.cols
        for x in xrange(len(ret[y])):
            #get random action
            rand = np.random.random_sample()
            d_r = 1.0 / len(action_set)
            cur_prob = 0
            ret_action = 'u'
            for a in action_set:
                cur_prob += d_r
                if rand < cur_prob:
                    ret_action = a
                    break

            ret[y][x] = ret_action
    return ret

def run_bfs(path, actions, prob_set):
    '''
    Function that runs BFS. Displays the path using a deterministic robot
        as well as a stochastic robot

    path - path to the map file
    actions - which action set to use (defined below)
    prob_set - which probability set to use (defined below)

     Available action sets are:
        'actions_1' - this allows for up, down, left, and right
        'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right

    Available Probability sets are:
        'prob_1' - 80% chance of going to target, 10% chance of going perpendicular
        'prob_2' - 70% chance of going to target, 10% chance of going perpendicular, 5% chance of going between
    '''
    input = [None] * 5
    input[0] = None
    input[1] = path
    input[2] = actions
    input[3] = prob_set
    input[4] = 'bfs'
    main(input)

def run_value_iteration(path, actions, prob_set, discount, base_reward, goal_reward, corner_reward = 0, use_corner = False):       
    '''
    Function that runs value iteration. Displays the map of values as well as a second map of actions

    path - path to the map file
    actions - which action set to use (defined below)
    prob_set - which probability set to use (defined below)
    discount - the discount factor (lambda)
    base_reward - the reward used on normal tiles
    goal_reward - the reward used on the goal tile
    corner_reward (optional) - the reward to be used on corners. Defaults to base_reward
    use_corner (optional) - whether or not to use the corner reward for corners. Default false

     Available action sets are:
        'actions_1' - this allows for up, down, left, and right
        'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right

    Available Probability sets are:
        'prob_1' - 80% chance of going to target, 10% chance of going perpendicular
        'prob_2' - 70% chance of going to target, 10% chance of going perpendicular, 5% chance of going between
        'prob_3' - 60% chance of going to target, 20% chance of going perpendicular in each direction
        'prob_4' - 80% chance of going to target, 10% chance of going in between it and perpendiculars
    '''    
    input = []
    if use_corner:
        input = [None] * 9
        input[8] = str(corner_reward)
    else:
        input = [None] * 8
    input[0] = None
    input[1] = str(path)
    input[2] = str(actions)
    input[3] = str(prob_set)
    input[4] = 'value_iteration'
    input[5] = str(discount)
    input[6] = str(base_reward)
    input[7] = str(goal_reward)
    main(input)

def run_policy_iteration(path, actions, prob_set, discount, base_reward, goal_reward, action, corner_reward = 0, use_corner = False):    
    '''
    Function that runs policy iteration. Displays the map of values as well as a second map of actions

    path - path to the map file
    actions - which action set to use (defined below)
    prob_set - which probability set to use (defined below)
    discount - the discount factor (lambda)
    base_reward - the reward used on normal tiles
    goal_reward - the reward used on the goal tile
    action - which (string) action to default to for the initial policy (defined below)
    corner_reward (optional) - the reward to be used on corners. Defaults to base_reward
    use_corner (optional) - whether or not to use the corner reward for corners. Default false

     Available action sets are:
        'actions_1' - this allows for up, down, left, and right
        'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right

    Available Probability sets are:
        'prob_1' - 80% chance of going to target, 10% chance of going perpendicular
        'prob_2' - 70% chance of going to target, 10% chance of going perpendicular, 5% chance of going between
        'prob_3' - 60% chance of going to target, 20% chance of going perpendicular in each direction
        'prob_4' - 80% chance of going to target, 10% chance of going in between it and perpendiculars

    Available action strings are:
        'u' - Initial map filled with the up action
        'l' - Initial map filled with the left action
        'r' - Initial map filled with the right action
        'd' - Initial map filled with the down action
        'ne' - Initial map filled with the up-right action
        'nw' - Initial map filled with the up-left action
        'se' - Initial map filled with the down-right action
        'sw' - Initial map filled with the down-left action
        'random' - Initial map filled with the random actions chosen from the chosen action set
    '''           
    input = []
    if use_corner:
        input = [None] * 10
        input[9] = str(corner_reward)
    else:
        input = [None] * 9
    input[0] = None
    input[1] = str(path)
    input[2] = str(actions)
    input[3] = str(prob_set)
    input[4] = 'policy_iteration'
    input[5] = str(discount)
    input[6] = str(base_reward)
    input[7] = str(goal_reward)
    input[8] = str(action)
    main(input)

def help():
    print('Must pass a minimum of 4 parameters: [file path] [actions] [probability_set] [algorithm] [algorithm parameters*]')
    print('    *[file path] - The path to the file representing the map')
    print('    *[actions] - Which action set to be used. Available action sets:')
    print('        *actions_1 - up, down, left, right')
    print('        *actions_2 - up, down, left, right, up-left, up-right, down-left, down-right')
    print('    *[probability_set] - which set of probabilities to use')
    print('        *prob_1 - 80% towards target, 10% perpendicular in each direction')
    print('        *prob_2 - 70% towards target, 10% perpendicular in each direction, 5% between toward and perpendicular')
    print('        *prob_3 - 60% towards target, 20% perpendicular in each direction')
    print('        *prob_4 - 80% chance of going to target, 10% chance of going in between it and perpendiculars')
    print('    *[algorithm] - Which algorithm is to be used. This can be: bfs, policy_iteration, value_iteration')
    print('            *bfs - breadth first search. Shows resulting map with deterministic robot, then resulting path with stochastic robot')
    print('            *value_iteration - performs value iteration. Prints final map of values and another map of actions')
    print('            *policy_iteration - performs policy iteration. Prints final map of values and another map of actions')
    print('    *[algorithm parameters*] - Whatever other parameters are need for the algorithm. Only applies to value_iteration and policy_iteration')
    print('    *For value iteration: [algorithm parameters] = [discount] [base reward] [goal reward] [corner reward (optional)]')
    print('    *For policy iteration: [algorithm parameters] = [discount] [base reward] [goal reward] [action] [corner reward (optional)]')
    print('            *discount - the discount factor (lambda)')
    print('            *base reward - the base reward to be used for normal tiles')
    print('            *goal reward - the goal\'s reward')
    print('            *corner reward - optional parameter to be used as the reward for corners. Use nothing if you just want this = base reward')
    print('            *actions - which set of actions to fill the initial policy grid with. Options are:')
    print('                 u - fill grid with up action')
    print('                 d - fill grid with down action')
    print('                 l - fill grid with left action')
    print('                 r - fill grid with right action')
    print('                 ne - fill grid with up-right action')
    print('                 nw - fill grid with up-left action')
    print('                 se - fill grid with down-right action')
    print('                 sw - fill grid with down-left action')
    print('                 random - fill grid with random actions from the given action set')    
    print('')
    print('Examples using command-line:')
    print('    python planning_with_uncertainty.py Tests/map1.txt actions_2 prob_1 bfs')
    print('    python planning_with_uncertainty.py Tests/map0.txt actions_1 prob_2 value_iteration 0.8 0.0 10.0')
    print('    python planning_with_uncertainty.py Tests/map2.txt actions_2 prob_1 policy_iteration 0.8 0.0 10.0 random -10.0')
    print('')
    print('Examples using API:')
    print('    planning_with_uncertainty.run_bfs(\'./Tests/map1.txt\', \'actions_2\', \'prob_1\')')
    print('    planning_with_uncertainty.run_value_iteration(\'./Tests/map0.txt\', \'actions_1\', \'prob_2\', 0.8, 0.0, 10.0, False)')
    print('    planning_with_uncertainty.run_policy_iteration(\'./Tests/map1.txt\', \'actions_2\', \'prob_1\', 0.8, 0.0, 10.0, random, -10.0, True)')
    
def main(argv):
    actions = []

    if len(argv) < 5:
        print('Too few arguments passed')
        print('')
        help()
        return

    if argv[2] == 'actions_1':
        actions = _ACTIONS
    elif argv[2] == 'actions_2':
        actions = _ACTIONS_2
    else:
        print('Action \'' + argv[2] + '\' could not be interpreted. Should be actions_1 or actions_2')
        print('')
        help()
        return

    probs = _PROBS
    if argv[3] == 'prob_1':
        probs = _PROBS
    elif argv[3] == 'prob_2':
        probs = _PROBS_2
    elif argv[3] == 'prob_3':
        probs = _PROBS_3
    elif argv[3] == 'prob_4':
        probs = _PROBS_4
    else:
        print('Could not interpret prob. set: ' + str(argv[3]))
        print('')
        help()
        return

    print('Reading map...')
    map = GridMap(argv[1])
    path = ([],{})

    print('Performing ' + argv[4] + '...')

    #determine which algorithm to run
    if argv[4] == 'bfs':
        print('Performing BFS...')
        path = bfs(map.init_pos, map.transition, map.is_goal, actions)
        if path == None:
            print('No path could be found. Exiting')
            return 

        map.display_map(path[0][0], path[1])
        path_stochastic = backpath_stochastic(map.init_pos, path[0][1], probs, map.transition)
        map.display_map(path_stochastic[0], path[1])

    elif argv[4] == 'value_iteration':
        if len(argv) < 8:
            print('Not enough arguments given')
            print('')
            help()
            return

        discount = float(argv[5])
        br = float(argv[6])
        gr = float(argv[7])
        cr = 0
        uc = False
        if len(argv) == 9:
            cr = float(argv[8])
            uc = True

        print('Discount: ' + str(discount))
        print('Base Reward: ' + str(br))
        print('Goal Reward: ' + str(gr))
        print('Using corner rewards: ' + str(uc) + ' with value of: ' + str(cr))

        v_map, iterations = value_iteration(map, map.transition, discount, actions, probs, br, gr, cr, uc)
        print('Number of iterations: ' + str(iterations))
        map.display_values(v_map)
        map.display_states(v_map)
        return

    elif argv[4] == 'policy_iteration':
        if len(argv) < 9:
            print('Not enough arguments given')
            print('')
            help()
            return

        discount = float(argv[5])
        br = float(argv[6])
        gr = float(argv[7])
        action = argv[8]
        cr = 0
        uc = False
        if len(argv) == 10:
            cr = float(argv[9])
            uc = True

        print('Discount: ' + str(discount))
        print('Base Reward: ' + str(br))
        print('Goal Reward: ' + str(gr))
        print('Action: ' + action)
        print('Using corner rewards: ' + str(uc) + ' with value of: ' + str(cr))
        #check action
        v_map = []
        iterations = 0
        if action == 'u' or action == 'd' or action == 'l' or action == 'r' or action == 'ne' or action == 'nw' or action == 'se' or action == 'sw':
            v_map, iterations = policy_iteration(map, map.transition, make_grid_action(map, action), discount, actions, probs, br, gr, cr, uc)
        elif action == 'random':
            v_map, iterations = policy_iteration(map, map.transition, make_grid_action_random(map, actions), discount, actions, probs, br, gr, cr, uc)
        else:
            print('Invalid action given: ' + action)
            print('')
            help()
            return
            
        print('Number of iterations: ' + str(iterations))
        map.display_values(v_map)
        map.display_states(v_map)
        return

    else:
        print('Algorithm: \'' + argv[4] + '\' is not recognized')
        print('')
        help()
        return    

if __name__ == "__main__":
    main(sys.argv)
