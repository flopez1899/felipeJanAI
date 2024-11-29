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
from contest.game import Directions
from contest.util import nearest_point


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
    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:  # Garantiza que haya al menos una acción válida
            return Directions.STOP

        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_state(self.index).get_position()
        carried_food = game_state.get_agent_state(self.index).num_carrying

        # Detectar agentes defensivos enemigos visibles
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Evitar fantasmas cercanos
        danger_threshold = 3  # Distancia crítica para decidir si recolectar más puntos
        ghost_positions = [ghost.get_position() for ghost in ghosts]
        safe_actions = actions[:]
        for ghost_pos in ghost_positions:
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)
                if self.get_maze_distance(pos, ghost_pos) < 2 and action in safe_actions:
                    safe_actions.remove(action)

        # Si no hay acciones seguras, priorizamos alejarnos de los fantasmas
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

        # 1. Si lleva puntos y hay fantasmas cerca, regresa a casa
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

    # 2. Si hay alimentos cercanos y los fantasmas están lejos, recolecta más puntos
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

        # Antes de regresar a casa, verifica si hay comida cercana y fantasmas lejanos
            next_pos = self.get_successor(game_state, best_action).get_agent_position(self.index)
            next_food_distances = [self.get_maze_distance(next_pos, food) for food in food_list if food != next_pos]
            if next_food_distances:
                next_min_distance = min(next_food_distances)
                close_ghosts_next = any(self.get_maze_distance(next_pos, ghost_pos) <= danger_threshold for ghost_pos in ghost_positions)
            if next_min_distance <= 2 and not close_ghosts_next:
                return best_action

            return best_action if best_action else random.choice(safe_actions)

    # 3. Si no queda comida o hay riesgo, realizar una acción válida
        return random.choice(safe_actions if safe_actions else actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        # return features
        
        return features


    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -10, 'distance_to_home': -5}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
