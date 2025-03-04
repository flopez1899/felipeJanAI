# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import contest.util as util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from contest.game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # util.raise_not_defined()
    
    """
    DESCRIPTION:
    We followed the pseudocode from the course slides to implement DFS.
    We store the frontier in a stack and the visited nodes in a set. While we have nodes on
    the frontier, we pop the last node and check if it is the goal. If it is, we return the
    path. If it is not, we add the node to the visited set and expand it. We add the 
    successors to the frontier if they have not been visited yet. The process continues until
    we find the goal or the frontier is empty.
    
    """
    
    # Stack to store (state, actions to reach state)
    frontier = util.Stack()
    
    # Initialize the stack with the start state, no actions, and zero cost
    start_state = problem.get_start_state()
    frontier.push(SearchNode(None, (start_state, None, 0)))
    
    # Set to track visited nodes
    visited = set()

    while not frontier.is_empty():
        current_node = frontier.pop()
        current_state = current_node.state

        # If current state is the goal, return the path of actions to get there
        if problem.is_goal_state(current_state):
            return current_node.get_path()

        # If we haven't visited this node, expand it
        if current_state not in visited:
            visited.add(current_state)
            
            # Get successors (state, action, cost)
            for successor, action, step_cost in problem.get_successors(current_state):
                if successor not in visited:
                    # Create a new node and push it onto the stack
                    frontier.push(SearchNode(current_node, (successor, action, step_cost)))

    # Return an empty list if no solution is found
    return []
        

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raise_not_defined()
    
    """
    DESCRIPTION:
    The algorithm has the same structure as the DFS, but we use a queue instead of a stack.
    """
    
    # Stack to store (state, actions to reach state)
    frontier = util.Queue()
    
    # Initialize the stack with the start state, no actions, and zero cost
    start_state = problem.get_start_state()
    frontier.push(SearchNode(None, (start_state, None, 0)))
    
    # Set to track visited nodes
    visited = set()

    while not frontier.is_empty():
        current_node = frontier.pop()
        current_state = current_node.state

        # If current state is the goal, return the path of actions to get there
        if problem.is_goal_state(current_state):
            return current_node.get_path()

        # If we haven't visited this node, expand it
        if current_state not in visited:
            visited.add(current_state)
            
            # Get successors (state, action, cost)
            for successor, action, step_cost in problem.get_successors(current_state):
                if successor not in visited:
                    # Create a new node and push it onto the stack
                    frontier.push(SearchNode(current_node, (successor, action, step_cost)))

    # Return an empty list if no solution is found
    return []
    

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    # util.raise_not_defined()
    
    """
    DESCRIPTION: 
    For UCS, we use a priority queue to store the nodes. The algorithm is similar to the previous,
    but the function to calculate the priority is different. Now, when we add a node to the frontier,
    we also add the cost of the path to the node (cost of the parent + cost of the step).
    
    """
    
    # Stack to store (state, actions to reach state)
    frontier = util.PriorityQueue()
    
    # Initialize the stack with the start state, no actions, zero cost and priority 0
    start_state = problem.get_start_state()
    frontier.push(SearchNode(None, (start_state, None, 0)),0)
    
    # Set to track visited nodes
    visited = set()

    while not frontier.is_empty():
        current_node = frontier.pop()
        current_state = current_node.state

        # If current state is the goal, return the path of actions to get there
        if problem.is_goal_state(current_state):
            return current_node.get_path()

        # If we haven't visited this node, expand it
        if current_state not in visited:
            visited.add(current_state)
            
            # Get successors (state, action, cost)
            for successor, action, step_cost in problem.get_successors(current_state):
                if successor not in visited:
                    # Create a new node and push it onto the stack with new priority
                    frontier.push(SearchNode(current_node, (successor, action, step_cost)), current_node.cost + step_cost)

    # Return an empty list if no solution is found
    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raise_not_defined()
    
    """
    DESCRIPTION:
    The algorithm is similar to UCS, but now we also consider the heuristic cost when adding
    a node to the frontier. The priority is the cost of the path to the node plus its heuristic
    cost to the goal.
    """
        
    # Stack to store (state, actions to reach state)
    frontier = util.PriorityQueue()
    
    # Initialize the stack with the start state, no actions, and zero cost
    start_state = problem.get_start_state()
    heuristic_start_cost = heuristic(start_state, problem)
    frontier.push(SearchNode(None, (start_state, None, 0)), heuristic_start_cost)
    
    # Set to track visited nodes
    visited = set()

    while not frontier.is_empty():
        current_node = frontier.pop()
        current_state = current_node.state

        # If current state is the goal, return the path of actions to get there
        if problem.is_goal_state(current_state):
            return current_node.get_path()

        # If we haven't visited this node, expand it
        if current_state not in visited:
            visited.add(current_state)
            
            # Get successors (state, action, cost)
            for successor, action, step_cost in problem.get_successors(current_state):
                if successor not in visited:
                    # Create a new node and push it onto the stack
                    heuristic_cost = heuristic(successor, problem)
                    frontier.push(SearchNode(current_node, (successor, action, step_cost)), current_node.cost + step_cost + heuristic_cost)

    # Return an empty list if no solution is found
    return []
    
    

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
