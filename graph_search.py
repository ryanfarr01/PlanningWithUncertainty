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
_ACTION_COST   = {'u': 1, 'd': 1, 'l': 1, 'r': 1}
_ACTIONS_2     = ['u','d','l','r','ne','nw','sw','se']
_ACTION_2_COST = {'u': 1, 'd': 1, 'l': 1, 'r': 1, 'ne': 1.5, 'nw': 1.5, 'sw': 1.5, 'se': 1.5}
_X = 1
_Y = 0
_GOAL_COLOR    = 0.75
_INIT_COLOR    = 0.25
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

    def transition(self, s, a):
        '''
        Transition function for the current grid map.

        s - tuple describing the state as (row, col) position on the grid.
        a - the action to be performed from state s

        returns - s_prime, the state transitioned to by taking action a in state s.
        If the action is not valid (e.g. moves off the grid or into an obstacle)
        returns the current state.
        '''
        new_pos = list(s[:])
        # Ensure action stays on the board
        if a == 'u':
            if s[_Y] > 0:
                new_pos[_Y] -= 1
        elif a == 'd':
            if s[_Y] < self.rows - 1:
                new_pos[_Y] += 1
        elif a == 'l':
            if s[_X] > 0:
                new_pos[_X] -= 1
        elif a == 'r':
            if s[_X] < self.cols - 1:
                new_pos[_X] += 1
        elif a == 'ne':
            if s[_X] < self.cols - 1 and s[_Y] > 0:
                new_pos[_X] += 1
                new_pos[_Y] -= 1
        elif a == 'nw':
            if s[_X] > 0 and s[_Y] > 0:
                new_pos[_X] -= 1
                new_pos[_Y] -= 1
        elif a == 'se':
            if s[_X] < self.cols - 1 and s[_Y] < self.rows - 1:
                new_pos[_X] += 1
                new_pos[_Y] += 1
        elif a == 'sw':
            if s[_X] > 0 and s[_Y] < self.rows - 1:
                new_pos[_X] -= 1
                new_pos[_Y] += 1
        else:
            print 'Unknown action:', str(a)

        # Test if new position is clear
        if self.occupancy_grid[new_pos[0], new_pos[1]]:
            s_prime = tuple(s)
        else:
            s_prime = tuple(new_pos)
        return s_prime

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
        display_grid[self.goal] = _GOAL_COLOR

        # Plot display grid for visualization
        imgplot = plotter.imshow(display_grid)
        # Set interpolation to nearest to create sharp boundaries
        imgplot.set_interpolation('nearest')
        # Set color map to diverging style for contrast
        imgplot.set_cmap('spectral')
        plotter.show()

    def uninformed_heuristic(self, s):
        '''
        Example of how a heuristic may be provided. This one is admissable, but dumb.

        s - tuple describing the state as (row, col) position on the grid.

        returns - floating point estimate of the cost to the goal from state s
        '''
        return 0.0

    def euclidian_distance_heuristic(self, s):
        return math.sqrt(math.pow(s[_X] - self.goal[_X], 2) + math.pow(s[_Y] - self.goal[_Y], 2))

    def manhattan_distance_heuristic(self, s):
        return abs(s[_X] - self.goal[_X]) + abs(s[_Y] - self.goal[_Y])

    def chebyshev_distance_heuristic(self, s):
        return min(abs(s[_X] - self.goal[_X]), abs(s[_Y] - self.goal[_Y]))


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

class PriorityQ:
    '''
    Priority queue implementation with quick access for membership testing
    Setup currently to only with the SearchNode class
    '''
    def __init__(self):
        '''
        Initialize an empty priority queue
        '''
        self.l = [] # list storing the priority q
        self.s = set() # set for fast membership testing

    def __contains__(self, x):
        '''
        Test if x is in the queue
        '''
        return x in self.s

    def push(self, x, cost):
        '''
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        '''
        if x.state in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(x.state)

    def pop(self):
        '''
        Get the value and remove the lowest cost element from the queue
        '''
        x = heapq.heappop(self.l)
        self.s.remove(x[1].state)
        return x[1]

    def peak(self):
        '''
        Get the value of the lowest cost element in the priority queue
        '''
        x = self.l[0]
        return x[1]

    def __len__(self):
        '''
        Return the number of elements in the queue
        '''
        return len(self.l)

    def replace(self, x, new_cost):
        '''
        Removes element x from the q and replaces it with x with the new_cost
        '''
        for y in self.l:
            if x.state == y[1].state:
                self.l.remove(y)
                self.s.remove(y[1].state)
                break
        heapq.heapify(self.l)
        self.push(x, new_cost)

    def get_cost(self, x):
        '''
        Return the cost for the search node with state x.state
        '''
        for y in self.l:
            if x.state == y[1].state:
                return y[0]

    def __str__(self):
        '''
        Return a string of the contents of the list
        '''
        return str(self.l)

def dfs(init_state, f, is_goal, actions):
    '''
    Perform depth first search on a grid map.

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
        n_i = frontier.pop()
        if n_i.state not in visited:
            visited.add(n_i.state)
            if is_goal(n_i.state):
                return(backpath(n_i), visited)
            else:
                for a in actions:
                    s_prime = f(n_i.state, a)
                    n_prime = SearchNode(s_prime, actions, n_i, a)
                    frontier.append(n_prime)
    
    #If we get here, the goal was never reached
    return None

def iterative_deepening(init_state, f, is_goal, actions, max_depth):
    '''
    iterative_deepening driver function

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent
    max_depth - the maximum allowed depth to search

    returns - ((path, action_path), visited) or None if no path can be found
    '''
    visited = set()
    for i in range(0, max_depth + 1):
        result = iterative_deepening_search(init_state, f, is_goal, actions, i, visited)
        if result != None:
            return result

    return None

def iterative_deepening_search(init_state, f, is_goal, actions, max_depth, visited):
    '''
    Performs iterative deepening depth-first search

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent
    max_depth - the maximum allowed depth to search before giving up
    visited - the set of visited nodes, may not be empty

    returns - ((path, action_path), visited) or None if no path can be found
    '''

    frontier = [] #The search stack
    n0 = SearchNode(init_state, actions)
    frontier.append((n0, 0))
    while len(frontier) > 0:
        n_i = frontier.pop()
        node = n_i[0]
        depth = n_i[1]
        if (node.state, depth, max_depth) not in visited and depth <= max_depth:
            visited.add((node.state, depth, max_depth))
            if is_goal(node.state):
                ret_visited = set()
                for t in visited:
                    ret_visited.add(t[0])
                return(backpath(node), ret_visited)
            else:
                for a in actions:
                    s_prime = f(node.state, a)
                    n_prime = SearchNode(s_prime, actions, node, a)
                    frontier.append((n_prime, depth + 1))

    return None

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
                    s_prime = f(n_i.state, a)
                    n_prime = SearchNode(s_prime, actions, n_i, a)
                    frontier.append(n_prime)
    
    #If we get here, the goal was never reached
    return None

def uniform_cost_search(init_state, f, is_goal, actions, action_cost):
    '''
    init_state - value of the initial state
    f - transition function takes input state (s), action (a), returns s_prime = f(s, a)
    is_goal - takes state as input returns true if it is a goal
    actions - list of possible actions available
    action_cost - mapping of actions to their cost
    '''
    frontier = PriorityQ()
    n0 = SearchNode(init_state, actions)
    visited = set()
    frontier.push(n0, 0)
    while len(frontier) > 0:
        n_i = frontier.pop()
        if n_i.state not in visited:
            visited.add(n_i.state)
            if is_goal(n_i.state):
                return(backpath(n_i), visited)
            else:
                for a in actions:
                    s_prime = f(n_i.state, a)
                    n_prime = SearchNode(s_prime, actions, n_i, a, n_i.cost + action_cost[a])
                    if (n_prime.state not in frontier) or (n_prime.cost < frontier.get_cost(n_prime)):
                        frontier.push(n_prime, n_prime.cost)
    return None

def a_star_search(init_state, f, is_goal, actions, action_cost, h):
    '''
    init_state - value of the initial state
    f - transition function takes input state (s), action (a), returns s_prime = f(s, a)
        returns s if action is not valid
    is_goal - takes state as input returns true if it is a goal state
        actions - list of actions available
    h - heuristic function, takes input s and returns estimated cost to goal
        (note h will also need access to the map, so should be a member function of GridMap)
    '''
    frontier = PriorityQ()
    n0 = SearchNode(init_state, actions)
    visited = set()
    frontier.push(n0, 0)
    while len(frontier) > 0:
        n_i = frontier.pop()
        if n_i.state not in visited:
            visited.add(n_i.state)
            if is_goal(n_i.state):
                return(backpath(n_i), visited)
            else:
                for a in actions:
                    s_prime = f(n_i.state, a)
                    n_prime = SearchNode(s_prime, actions, n_i, a, n_i.cost + action_cost[a])
                    tot_cost = n_prime.cost + h(n_prime.state)
                    if (n_prime.state not in frontier) or (tot_cost < frontier.get_cost(n_prime)):
                        frontier.push(n_prime, tot_cost)
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
    

def get_heuristic(map, h):
    if h == 'uninformed':
        return map.uninformed_heuristic
    elif h == 'euclidian':
        return map.euclidian_distance_heuristic
    elif h == 'manhattan':
        return map.manhattan_distance_heuristic
    elif h == 'chebyshev':
        return map.chebyshev_distance_heuristic

    print('Could not interpret heuristic \'' + h + '\'. Accepted: uninformed, euclidian, manhattan, chebyshev.')
    print('')
    help()
    return -1

def main(argv):
    if len(argv) != 4 and len(argv) != 5:
        help()
        return
    
    actions = []
    actions_cost = {}
    if argv[2] == 'actions_1':
        actions = _ACTIONS
        actions_cost = _ACTION_COST
    elif argv[2] == 'actions_2':
        actions = _ACTIONS_2
        actions_cost = _ACTION_2_COST
    else:
        print('Action \'' + argv[2] + '\' could not be interpreted. Should be actions_1 or actions_2')
        print('')
        help()

    print('Creating map...')
    map = GridMap(argv[1])
    path = ([],{})

    if argv[3] == 'dfs':
        print('Performing DFS...')
        path = dfs(map.init_pos, map.transition, map.is_goal, actions)
    elif argv[3] == 'iterative_deepening':
        print('Performing iterative deepening...')
        path = iterative_deepening(map.init_pos, map.transition, map.is_goal, actions, (map.rows + 1)*(map.cols + 1))
    elif argv[3] == 'bfs':
        print('Performing BFS...')
        path = bfs(map.init_pos, map.transition, map.is_goal, actions)
    elif argv[3] == 'uniform':
        print('Performing uniform cost search...')
        path = uniform_cost_search(map.init_pos, map.transition, map.is_goal, actions, actions_cost)
    elif argv[3] == 'a_star':
        print('Performing A* pathfinding with heuristic: ' + argv[4])
        heuristic = get_heuristic(map, argv[4])
        if heuristic == -1:
            return 
        path = a_star_search(map.init_pos, map.transition, map.is_goal, actions, actions_cost, heuristic)
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

if __name__ == "__main__":
    main(sys.argv)