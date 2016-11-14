#!/usr/bin/env python
'''
Package providing helper classes and functions for performing graph search operations for planning.

Author: Ryan Farr
Date: 9/8/2016
File: graph_search.py - file contains all data structures and functions necessary to run
    depth-first search, iterative deepening, breadth-first search, uniform cost search,
    and A* search. Can be used by directly passing arguments through the commandline or 
    by using the run_algorithm function after importing graph_search into another python
    file.

Available action sets are:
    'actions_1' - this allows for up, down, left, and right
    'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right

Available algorithms are:
    'dfs'                 - Depth-First Search
    'iterative_deepening' - Iterative Deepning Depth-First Search
    'bfs'                 - Breadth-First Search
    'uniform'             - Uniform Cost Search
    'a_star'              - A* Search

Available heuristics are:
    'uninformed' - Always returns 0
    'euclidian'  - Returns the Euclidian distance to the goal
    'manhattan'  - Returns the Manhattan distance to the goal
    'chebyshev'  - Returns the chebyshev distance to the goal
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
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)
        min_val = 10000000
        max_val = -10000000

        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
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
                    display_grid[(y, x)] = _INIT_COLOR + ((map[y][x][0] - min_val)/max_val)

            display_grid[self.goal] = _GOAL_COLOR
            imgplot = plotter.imshow(display_grid)
            imgplot.set_interpolation('nearest')
            imgplot.set_cmap('spectral')
            plotter.show()

    def display_values(self, map):
        '''
        Visualize the map read in. Optionally display the resulting plan and visisted nodes

        path - a list of tuples describing the path take from init to goal
        '''
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)

        min_val = 10000000
        max_val = -10000000
        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                plotter.text(x, y, "%.2f" % map[y][x][0])
                if map[y][x][0] > max_val:
                    max_val = map[y][x][0]
                if map[y][x][0] < min_val:
                    min_val = map[y][x][0]

        for y in xrange(len(map)):
            for x in xrange(len(map[y])):
                if self.occupancy_grid[y][x]:
                    display_grid[(y, x)] = _BLACK
                else:
                    display_grid[(y,x)] = _INIT_COLOR + ((map[y][x][0] - min_val)/max_val)
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
    reward_grid = [[base_reward for i in xrange(0, map.cols)] for i in xrange(0, map.rows)]
    reward_grid[map.goal[_Y]][map.goal[_X]] = goal_reward
    if use_corners:
        reward_grid[0][0] = corner_reward
        reward_grid[0][map.cols-1] = corner_reward
        grireward_grid[map.rows-1][0] = corner_reward
        grireward_grid[map.rows-1][map.cols-1] = corner_reward
    
    # the value grid is the same as the reward grid initially
    #print('map rows: ' + str(map.rows))
    #print('map cols: ' + str(map.cols))
    value_grid = [None] * map.rows
    for y in xrange(len(value_grid)):
        value_grid[y] = [0] * map.cols
        for x in xrange(len(value_grid[y])):
            r = reward_grid[y][x]
            value_grid[y][x] = (r, r) #g[0] = current, g[1] = previous value
    print('value_grid dimensions: ' + str(len(value_grid)) + 'x' + str(len(value_grid[0])))

    #Iterate to get value iteration
    needs_iteration = True
    iter = 0
    while(needs_iteration):
        iter += 1
        needs_iteration = False
        for y in xrange(len(value_grid)):
            for x in xrange(len(value_grid[y])):
                #print('')
                #print('at x and y: ' + str(x) + ' ' + str(y))
                #print('--------------------------')
                #get max action value and set as value in grid
                max_val = -sys.maxint - 1 #get minimum value
                for a in action_set:
                    #print('')
                    cur_val = 0
                    a_set = t((y, x), a, probs, False)
                    for s, prob in a_set:
                        #print('prob: ' + str(prob))
                        #print('reward: ' + str(reward_grid[s[0]][s[1]]))
                        #print('discount: ' + str(discount))
                        #print('s: ' + str(s[0]) + ' ' + str(s[1]))
                        #print('value_grid: ' + str(len(value_grid)) + 'x' + str(len(value_grid[0])))
                        s_x = s[_X]
                        s_y = s[_Y]
                        reward = reward_grid[s_y][s_x]
                        value = value_grid[s_y][s_x][1]
                        #print('Looking at point: (' + str(s_x) + ', ' + str(s_y))
                        #print('value: ' + str(value))
                        #print('reward:' + str(reward))
                        #print('RHs: ' + str(discount*value))
                        #print('Total (before multiply): ' + str(reward + (discount*value)))
                        #print('Total: ' + str(prob*(reward+(discount*value))))
                        cur_val += prob * (reward + (discount * value))
                    
                    if cur_val > max_val:
                        max_val = cur_val
                temp_val = value_grid[y][x][0]
                value_grid[y][x] = (max_val, value_grid[y][x][1])
                if value_grid[y][x][0] != temp_val:
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
                    cur_val += prob * value_grid[s[0]][s[1]][0]
                
                if cur_val > max_val:
                    max_val = cur_val
                    best_action = a
            policy_grid[y][x] = (value_grid[y][x][0], best_action)

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

def run_algorithm(path, action, algorithm, heuristic = 'uninformed'):
    '''
    Function to run a given algorithm using a specified action set. Automatically shows the results
        Available action sets are:
            'actions_1' - this allows for up, down, left, and right
            'actions_2' - this allows for up, down, left, right, up left, up right, down left, and down right
        
        Available algorithms are:
            'dfs'                 - Depth-First Search
            'iterative_deepening' - Iterative Deepning Depth-First Search
            'bfs'                 - Breadth-First Search
            'uniform'             - Uniform Cost Search
            'a_star'              - A* Search

        Available heuristics are:
            'uninformed' - Always returns 0
            'euclidian'  - Returns the Euclidian distance to the goal
            'manhattan'  - Returns the Manhattan distance to the goal
            'chebyshev'  - Returns the chebyshev distance to the goal

        Example for depth-first search on map 0:
            run_algorithm('Tests/map0.txt', 'actions_1', 'dfs') 

        Example for A* search on map 2 using Manhattan distance as the heuristic:
            run_algorithm('tests/map2.txt', 'actions_2', 'a_star', 'manhattan')
    '''
    input = [None] * 5
    input[0] = None
    input[1] = path
    input[2] = action
    input[3] = algorithm
    input[4] = heuristic
    main(input)

def help():
    print('Must pass three or four arguments arguments: [file path] [actions] [algorithm] [optional: heuristic]')
    print('    *[file path] - The path to the file representing the map')
    print('    *[actions] - Which action set to be used. Available action sets:')
    print('        *actions_1 - up, down, left, right')
    print('        *actions_2 - up, down, left, right, up-left, up-right, down-left, down-right')
    print('    *[algorithm] - Which algorithm is to be used. This can be: dfs, iterative_deepening, bfs, uniform, or a_star')
    print('        If a_star is used, pass a third argument for the heurisitc to be used. Available heuristics:')
    print('            *uninformed - always returns 0')
    print('            *euclidian - returns the euclidian distance between the given point and the goal')
    print('            *manhattan - returns the manhattan distance between the given point and the goal')
    print('            *chebyshev - returns the chebyshev distance between the given point and the goal')
    print('')
    print('Example:')
    print('    python graph_search.py Tests/map1.txt actions_2 a_star euclidian')
    print('    python graph_search.py Tests/map2.txt actions_1 dfs')
    print('')
    print('Example using API:')
    print('    graph_search.run_algorithm(\'Tests/map1.txt\', \'actions_1\', \'bfs\')')
    
def main(argv):
    if len(argv) != 4 and len(argv) != 5:
        help()
        return
    
    actions = []
    if argv[2] == 'actions_1':
        actions = _ACTIONS
    elif argv[2] == 'actions_2':
        actions = _ACTIONS_2
    else:
        print('Action \'' + argv[2] + '\' could not be interpreted. Should be actions_1 or actions_2')
        print('')
        help()

    print('Creating map...')
    map = GridMap(argv[1])
    path = ([],{})
    print('Starting position is: ' + str(map.init_pos[_X]) + ' ' + str(map.init_pos[_Y]))

    if argv[3] == 'bfs':
        print('Performing BFS...')
        path = bfs(map.init_pos, map.transition, map.is_goal, actions)
    elif argv[3] == 'value_iteration':
        v_map, iterations = value_iteration(map, map.transition, 0.8, actions, _PROBS, base_reward = 0.0, goal_reward = 10.0, corner_reward = 0, use_corners = False)
        print('Number of iterations: ' + str(iterations))
        map.display_values(v_map)
        map.display_states(v_map)
        return
    else:
        print('Algorithm: \'' + argv[3] + '\' is not recognized')
        print('')
        help()
        return    

    print('Printing results...')
    if path == None:
        print('No path could be found. Exiting')
        return
    
    map.display_map(path[0][0], path[1])

    path_stochastic = backpath_stochastic(map.init_pos, path[0][1], _PROBS, map.transition)
    map.display_map(path_stochastic[0], path[1])

if __name__ == "__main__":
    main(sys.argv)
