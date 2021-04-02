
# MIT License
#
# Copyright (c) 2018 Blanyal D'Souza
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Classes for Monte Carlo Tree Search."""
import math

import numpy as np

from dopamine.discrete_domains.CGF import CFG
from copy import deepcopy


class TreeNode(object):
    """Represents a board state and stores statistics for actions at that state.
    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        action: A tuple(row, column) of the prior move of reaching this node.
        children: A list which stores child nodes.
        child_psas: A vector containing child probabilities.
        parent: A TreeNode representing the parent node.
    """

    def __init__(self, parent=None, action=None, psa=0.0, child_psas=[]):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.game_state="NO GAME STATE"
        self.Psa = psa
        self.action = action
        self.children = []
        self.child_psas = child_psas
        self.parent = parent

    def is_not_leaf(self):
        """Checks if a TreeNode is a leaf.
        Returns:
            A boolean value indicating if a TreeNode is a leaf.
        """
        if len(self.children) > 0:
            return True
        return False
    
    
    def get_child_from_action(self, action):
      action2=(1,((int)(action/3)),action%3)
      for child in self.children:
        if child.action==action2:
          return child
      shnob=pookipooky
      return None
    def select_child(self):
        """Selects a child node based on the AlphaZero PUCT formula.
        Returns:
            A child TreeNode which is the most promising according to PUCT.
        """
        c_puct = CFG.c_puct

        highest_uct = 0
        highest_index = 0

        # Select the child with the highest Q + U value
        for idx, child in enumerate(self.children):
            uct = child.Qsa + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / (1 + child.Nsa))
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def expand_node(self, game, psa_vector):
        """Expands the current node by adding valid moves as children.
        Args:
            game: An object containing the game state.
            psa_vector: A list containing move probabilities for each move.
        """
        self.child_psas = deepcopy(psa_vector)
        valid_moves = game.get_valid_moves(game.current_player)
        for idx, move in enumerate(valid_moves):
            if move[0] is not 0:
                action = deepcopy(move)
                self.add_child_node(parent=self, action=action,
                                    psa=psa_vector[idx])

    def add_child_node(self, parent, action, psa=0.0):
        """Creates and adds a child TreeNode to the current node.
        Args:
            parent: A TreeNode which is the parent of this node.
            action: A tuple(row, column) of the prior move to reach this node.
            psa: A float representing the raw move probability for this node.
        Returns:
            The newly created child TreeNode.
        """

        child_node = TreeNode(parent=parent, action=action, psa=psa)
        self.children.append(child_node)
        return child_node

    def back_prop(self, wsa, v):
        """Update the current node's statistics based on the game outcome.
        Args:
            wsa: A float representing the action value for this state.
            v: A float representing the network value of this state.
        """
        self.Nsa += 1
        self.Wsa=wsa+v
        self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch(object):
    """Represents a Monte Carlo Tree Search Algorithm.
    Attributes:
        root: A TreeNode representing the board state and its statistics.
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, net):
        """Initializes TreeNode with the TreeNode, board and neural network."""
        self.root = None
        self.game = None
        self.net = net

    def array_action_to_int_action(self, action):
        return action[1]*3+action[2]

    def int_action_to_array_action(self, action):
        return (1,((int)(action/3)),action%3)

    def search(self, game, node, temperature):
        """MCTS loop to get the best move which can be played at a given state.
        Args:
            game: An object containing the game state.
            node: A TreeNode representing the board state and its statistics.
            temperature: A float to control the level of exploration.
        Returns:
            A child node representing the best move to play at this state.
        """
        self.root = node
        self.game = game
        self.root.game_state=game._get_obs()
        for i in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Create a fresh clone for each loop.
            previous_game_state=None
            # Loop when node is not a leaf
            space=""
            while node.is_not_leaf():
                node = node.select_child()
                previous_game_state=game._get_obs()
                #print(space+"GAME STATE:", game._get_obs())
                #print(space+"ACTION BEFORE:", node.action)
                #print(space+"ACTION AFTER:", self.array_action_to_int_action(node.action) )
                game.play_action(self.array_action_to_int_action(node.action))
                #space=space+"   "

            # Get move probabilities and values from the network for this state.
            node.game_state=game._get_obs()            
            psa_vector, _ = self.net.predict(game._get_obs(), 0)
            if previous_game_state!=None:
                _, v=self.net.predict(previous_game_state, self.array_action_to_int_action(node.action))
            else:
                v=0
            # Add Dirichlet noise to the psa_vector of the root node.
            if node.parent is None:
                psa_vector = self.add_dirichlet_noise(game, psa_vector)

            valid_moves = game.get_valid_moves(game.current_player)
            
            
            for idx, move in enumerate(valid_moves):
                if move[0] is 0:
                    psa_vector[idx] = 0

            psa_vector_sum = sum(psa_vector)
            # Renormalize psa vector
            if psa_vector_sum > 0:
                psa_vector /= psa_vector_sum

            # Try expanding the current node.
            
            node.expand_node(game=game, psa_vector=psa_vector)

            game_over, wsa = game.check_game_over(game.current_player)
            # Back propagate node statistics up to the root node.
            while node is not None:
                wsa = -wsa
                v = -v
                node.back_prop(wsa, v)
                node = node.parent

        highest_nsa = 0
        highest_index = 0

        # Select the child's move using a temperature parameter.
        for idx, child in enumerate(self.root.children):
            temperature_exponent = int(1 / temperature)

            if child.Nsa ** temperature_exponent > highest_nsa:
                highest_nsa = child.Nsa ** temperature_exponent
                highest_index = idx

        return self.root.children[highest_index]

    def printTree(self, game, node, space=""):
        for child in node.children:
            print(space + "State:", child.game_state)
            print(space + "QSA:", child.Qsa)
            print(space + "NSA:", child.Nsa)
            self.printTree(game, child, space+"   ")

    def searchWrap(self, game, node, temperature):
        ret= self.search(game, node, temperature)
        ret.action=self.array_action_to_int_action(ret.action)
        return ret

    def add_dirichlet_noise(self, game, psa_vector):
        """Add Dirichlet noise to the psa_vector of the root node.
        This is for additional exploration.
        Args:
            game: An object containing the game state.
            psa_vector: A probability vector.
        Returns:
            A probability vector which has Dirichlet noise added to it.
        """
        dirichlet_input = [CFG.dirichlet_alpha for x in range(game.action_size)]

        dirichlet_list = np.random.dirichlet(dirichlet_input)
        noisy_psa_vector = []

        for idx, psa in enumerate(psa_vector):
            noisy_psa_vector.append(
                (1 - CFG.epsilon) * psa + CFG.epsilon * dirichlet_list[idx])
        return noisy_psa_vector