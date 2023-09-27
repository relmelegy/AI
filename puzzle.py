from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q

# Global variables for tracking statistics
nodes_expanded = 0
max_search_depth = 0
path_to_goal = []
cost_of_path = 0
search_depth = 0
start_time = 0

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        if self.blank_index < self.n:  # If the blank is in the top row, it can't move up
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index - self.n] = new_config[self.blank_index - self.n], new_config[self.blank_index]
            return PuzzleState(new_config, self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index >= (self.n * (self.n - 1)):  # If the blank is in the bottom row, it can't move down
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index + self.n] = new_config[self.blank_index + self.n], new_config[self.blank_index]
            return PuzzleState(new_config, self.n, parent=self, action="Down", cost=self.cost + 1)

    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index % self.n == 0:  # If the blank is in the leftmost column, it can't move left
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index - 1] = new_config[self.blank_index - 1], new_config[self.blank_index]
            return PuzzleState(new_config, self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        if (self.blank_index + 1) % self.n == 0:  # If the blank is in the rightmost column, it can't move right
            return None
        else:
            new_config = list(self.config)
            new_config[self.blank_index], new_config[self.blank_index + 1] = new_config[self.blank_index + 1], new_config[self.blank_index]
            return PuzzleState(new_config, self.n, parent=self, action="Right", cost=self.cost + 1)

    def expand(self):
        """ Generate the child nodes of this node """

        # Node has already been expanded
        if len(self.children) != 0:
            return self.children

        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
import resource

def writeOutput():
    global path_to_goal
    global cost_of_path
    global nodes_expanded
    global search_depth
    global max_search_depth
    global start_time
    import resource

    running_time = time.time() - start_time
    max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20)

    with open("output.txt", "w") as f:
        f.write("path to goal: {}\n".format(path_to_goal))
        f.write("cost of path: {}\n".format(cost_of_path))
        f.write("nodes expanded: {}\n".format(nodes_expanded))
        f.write("search depth: {}\n".format(search_depth))
        f.write("max search depth: {}\n".format(max_search_depth))
        f.write("running time: {:.8f}\n".format(running_time))
        f.write("max ram usage: {:.8f}\n".format(max_ram_usage))


def bfs_search(initial_state):
    """BFS search"""
    global nodes_expanded
    global max_search_depth
    global path_to_goal
    global cost_of_path
    global search_depth

    explored = set()
    frontier = Q.Queue()
    frontier.put(initial_state)

    while not frontier.empty():
        state = frontier.get()
        if test_goal(state):
            # Goal found
            current = state
            while current.parent:
                path_to_goal.insert(0, current.action)
                current = current.parent
            cost_of_path = state.cost
            search_depth = state.cost
            return state

        explored.add(tuple(state.config))
        children = state.expand()
        nodes_expanded += 1
        for child in children:
            if tuple(child.config) not in explored:
                frontier.put(child)
                if child.cost > max_search_depth:
                    max_search_depth = child.cost

def dfs_search(initial_state):
    """DFS search"""
    global nodes_expanded
    global max_search_depth
    global path_to_goal
    global cost_of_path
    global search_depth

    def is_in_frontier_or_explored(state, frontier_set, explored):
        """Helper function to check if a state is in frontier or explored"""
        return tuple(state.config) in frontier_set or tuple(state.config) in explored

    frontier = [initial_state]  # Using a list as a stack for DFS
    frontier_set = set([tuple(initial_state.config)])  # Set to quickly check if a state is in frontier
    explored = set()

    while frontier:
        state = frontier.pop()
        frontier_set.remove(tuple(state.config))  # Remove state from frontier_set
        explored.add(tuple(state.config))

        if test_goal(state):
            # Goal found
            current = state
            while current.parent:
                path_to_goal.insert(0, current.action)
                current = current.parent
            cost_of_path = state.cost
            search_depth = state.cost
            return state  # Return SUCCESS with the state

        children = reversed(state.expand())  # Reverse the order of children for correct DFS order
        nodes_expanded += 1
        for child in children:
            if not is_in_frontier_or_explored(child, frontier_set, explored):
                frontier.append(child)
                frontier_set.add(tuple(child.config))  # Add child to frontier_set
                if child.cost > max_search_depth:
                    max_search_depth = child.cost

    return



def A_star_search(initial_state):
    """A * search"""
    global nodes_expanded
    global max_search_depth
    global path_to_goal
    global cost_of_path
    global search_depth

    explored = set()
    start_node = (calculate_total_cost(initial_state) + initial_state.cost, 0, initial_state)  # Add a sequence number
    frontier = Q.PriorityQueue()
    frontier.put(start_node)

    seq_num = 1  # Start the sequence number at 1

    while not frontier.empty():
        _, _, state = frontier.get()  # Extract the state from the tuple
        if test_goal(state):
            # Goal found
            current = state
            while current.parent:
                path_to_goal.insert(0, current.action)
                current = current.parent
            cost_of_path = state.cost
            search_depth = state.cost
            return state

        explored.add(tuple(state.config))
        children = state.expand()
        nodes_expanded += 1
        for child in children:
            if tuple(child.config) not in explored:
                cost = child.cost + calculate_total_cost(child)
                frontier.put((cost, seq_num, child))  # Add the sequence number to the tuple
                seq_num += 1  # Increment the sequence number
                if child.cost > max_search_depth:
                    max_search_depth = child.cost



def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    cost = 0
    for idx, value in enumerate(state.config):
        if value != 0:  # We don't compute Manhattan distance for the blank tile
            cost += calculate_manhattan_dist(idx, value, state.n)
    return cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    if value == 0:  # Skip the blank tile
        return 0
    # Current row and column
    current_row = idx // n
    current_col = idx % n

    # Goal row and column
    goal_row = value // n
    goal_col = value % n

    return abs(current_row - goal_row) + abs(current_col - goal_col)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    goal_config = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return puzzle_state.config == goal_config

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    global start_time
    start_time  = time.time()

    if   search_mode == "bfs": goal_state = bfs_search(hard_state)
    elif search_mode == "dfs": goal_state = dfs_search(hard_state)
    elif search_mode == "ast": goal_state = A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        return
    if goal_state:
        writeOutput()

    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
