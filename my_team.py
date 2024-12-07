# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point

# Import A* search algorithm
from search import a_star_search


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_eaten_history = [] # list to track history for last moves

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Our offensive agent uses a variety of logical rules to determine the best offensive move.
    It classifies actions as safe or dangerous based on the distance to the closest ghost.
    """
    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions: return Directions.STOP # if no valid action, stop

        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_state(self.index).get_position()
        carried_food = game_state.get_agent_state(self.index).num_carrying

        # Detect defensive enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Avoid ghosts if they are too close
        danger_threshold = 3  # critical distance to decide if keep eating or run away
        ghost_positions = [ghost.get_position() for ghost in ghosts]
        safe_actions = actions[:]
        for ghost_pos in ghost_positions:
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)
                if self.get_maze_distance(pos, ghost_pos) < 2 and action in safe_actions:
                    safe_actions.remove(action)

        # If there are no safe actions, run away. Choose the action that maximizes the distance to the closest ghost.
        if not safe_actions:
            best_action = None
            max_distance = -float('inf')
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)
                distances = [self.get_maze_distance(pos, ghost_pos) for ghost_pos in ghost_positions]
                min_distance = min(distances) if distances else float('inf')
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_action = action
            return best_action if best_action else Directions.STOP

        # If we have points and there are ghosts nearby, return home
        if carried_food > 0:
            close_ghosts = any(self.get_maze_distance(my_pos, ghost_pos) <= danger_threshold for ghost_pos in ghost_positions)
            if close_ghosts:
                best_action = None
                best_distance = float('inf')
                for action in safe_actions:
                    successor = self.get_successor(game_state, action)
                    pos = successor.get_agent_position(self.index)
                    distance_home = self.get_maze_distance(pos, self.start)
                    if distance_home < best_distance:
                        best_distance = distance_home
                        best_action = action
                return best_action if best_action else random.choice(safe_actions)

        # If there is food nearby and ghosts are far away, collect more food
        if len(food_list) > 0:
            best_action = None
            best_distance = float('inf')
            for action in safe_actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)
                distances = [self.get_maze_distance(pos, food) for food in food_list]
                min_distance = min(distances) if distances else float('inf')
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_action = action

            # Before going back home, verify if there is food nearby and ghosts are far away
            next_pos = self.get_successor(game_state, best_action).get_agent_position(self.index)
            next_food_distances = [self.get_maze_distance(next_pos, food) for food in food_list if food != next_pos]
            if next_food_distances:
                next_min_distance = min(next_food_distances)
                close_ghosts_next = any(self.get_maze_distance(next_pos, ghost_pos) <= danger_threshold for ghost_pos in ghost_positions)
            if next_min_distance <= 2 and not close_ghosts_next:
                return best_action

            return best_action if best_action else random.choice(safe_actions)

        # If there is no food left or there is a risk, perform a valid action
        return random.choice(safe_actions if safe_actions else actions)


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Our defensive agent uses A* search to determine the best defensive move.
    It stays in the capsule until it sees an invader and goes after him.
    """

    def choose_action(self, game_state):
        """
        Uses A* to determine the best defensive move.
        """
        # Define the search problem
        problem = DefensiveSearchProblem(self, game_state)
        
        # Compute the path using A* search
        actions = a_star_search(problem)
        
        # Return the first action in the path, or stop if no path is found
        return actions[0] if actions else Directions.STOP

class DefensiveSearchProblem:
    """
    Search problem for the defensive agent.
    """

    def __init__(self, agent, game_state):
        self.agent = agent
        self.start_state = game_state.get_agent_position(agent.index)
        self.walls = game_state.get_walls()
        self.opponents = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)] # All opponent agents' states
        self.invaders = [a for a in self.opponents if a.is_pacman and a.get_position() is not None] # visible opponents
        self.capsules = game_state.get_capsules()
        self.food = agent.get_food_you_are_defending(game_state)
        self.goal_positions = [a.get_position() for a in self.invaders] or self.capsules # goal is invader position or capsule

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        return state in self.goal_positions

    def get_successors(self, state):
        """
        Function retrieved from PositionSeachProblem of search_agents.py, with now a cost of 1 for each action.
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                successors.append((next_state, action, 1))
        return successors

    def get_cost_of_actions(self, actions):
        """
        Function retrieved from PositionSeachProblem of search_agents.py.
        """
        if actions is None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x, y))
        return cost