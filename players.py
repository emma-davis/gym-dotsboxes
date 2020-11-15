"""
ALL PLAYER CLASSES INCLUDING:
 - USER PLAYER CLASS (UserPlayer)
 - DQN AGENT PLAYER CLASS (DQNPlayer)
 - GREEDY AGENT PLAYER CLASS (GreedyPlayer)
"""
import numpy as np
import random
from DQN import DQN_model
from gym_dotsboxes.environment import DotsBoxesEnv


class UserPlayer:

    # LETS USER PLAY GAME BY ENTERING THEIR ACTIONS AS NUMBERS
    def __init__(self, env):
        self.env = env

    # def set_environment(self, environment):
    #    self.env = environment

    def act(self):

        if self.env is None:
            raise ValueError("Environment has not been set.")

        # USER INPUTS CHOSEN VALID ACTION BY ENTERING NUMBER 0-23
        action = None
        while action not in self.env.available_actions:
            print("Valid actions: ", self.env.available_actions)
            action = int(input("Please choose an action:\n"))

        # APPLY ACTION TO ENVIRONMENT VIA STEP FUNCTION DEFINED IN
        # ENVIRONMENT CLASS
        self.env.step(action)

        return action


class GreedyPlayer:

    # GREEDY PLAYER ALWAYS COMPLETES A BOX IF IT CAN, AND AVOIDS
    # SACRIFICING BOXES (I.E. PLACING A THIRD LINE OF A BOX THAT
    # THE NEXT PLAYER MAY COMPLETE).
    def __init__(self, env):
        self.env = env

    def act(self):

        # USING BOARD CURRENT STATE, CHOOSE AN OPTION THAT EITHER
        # - COMPLETES A BOX
        # - FAILING THIS, RANDOM ACTION THAT DOES NOT CREATE THIRD
        # LINE IN SQUARE
        # - FAILING THIS, RANDOM ACTION (OUT OF ACTIONS THAT CREATE
        # THIRD LINE IN SQUARE)
        if self.env is None:
            raise ValueError("Environment has not been set.")

        greedy_actions = []
        current_state = self.env.state

        for action in self.env.available_actions:
            # TODO: CHECK LOGIC HERE
            greedy_actions.append(action)

        action = random.choice(greedy_actions)

        # APPLY ACTION TO ENVIRONMENT VIA STEP FUNCTION DEFINED IN
        # ENVIRONMENT CLASS
        # self.env.step(action)

        return action


class DQNPlayer:

    def __init__(self, env, alpha=0.001, epsilon=0.05, gamma=0.9):
        self.env = env
        self.learning = True
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.DQN = None

        self.past_state = None
        self.past_action = None

    # def set_environment(self, environment):
    #   self.env = environment

    def act(self):

        # CHOOSES ACTION BASED ON DQN MODEL HIGHEST Q-VALUE PREDICTED
        current_state = self.env.state
        matrix = self.get_3D_matrix(current_state)

        # IF RANDOM NUMBER LESS THAN EPSILON, CHOOSE RANDOM ACTION, OTHERWISE
        # PICK VALID ACTION THAT YIELDS HIGHEST PREDICTED Q VALUE FROM DQN MODEL
        if random.random() < self.epsilon and self.learning:
            action = np.random.choice(self.env.available_actions)
        else:
            q_values = self.DQN.predict(matrix)[0]
            max_q = q_values[self.env.available_actions].max()
            best_actions = np.where(q_values == max_q)[0]

            action = random.choice([action for action in best_actions if action in self.env.available_actions])

        self.past_state = matrix.copy()
        self.past_action = action

        return action

    def observe(self, state, reward):
        # OBSERVE STATE REWARD PAIR
        # TODO: NOT SURE WHAT THIS IS SUPPOSED TO DO, LEAVE BLANK FOR NOW
        return None

    def get_3D_matrix(self, state):

        # RESHAPES LIST OF BOARD INTO 3X3X4 MATRIX TO FEED INTO MODEL
        matrix = []

        # LIST OF STRINGS CORRESPONDING TO BOOLEAN POSITION ON DOTS GRID
        positions = ['n', 'n', 'n', 'w', 'ew', 'ew', 'e', 'ns', 'ns', 'ns', 'w', 'ew', 'ew', 'e',
                     'ns', 'ns', 'ns', 'w', 'ew', 'ew', 'e', 's', 's', 's']
        if (isinstance(state, list)) & (len(state) == 24):
            north_list = []
            east_list = []
            south_list = []
            west_list = []

            for i in range(len(state)):
                if positions[i] in ['n', 'ns']:
                    north_list.append(state[i])
                if positions[i] in ['e', 'ew']:
                    east_list.append(state[i])
                if positions[i] in ['s', 'ns']:
                    south_list.append(state[i])
                if positions[i] in ['w', 'ew']:
                    west_list.append(state[i])

            matrix = np.resize([north_list, east_list, south_list, west_list], (4, 3, 3))

        else:
            print("State does not conform to list of length 24. Please check this. State:",
                  state)

        # NEED TO ADD THIS TO CREATE EXTRA NONE DIMENSION FOR FEEDING INTO MODEL
        matrix = matrix[None, :]
        print(matrix)
        return matrix

    # TODO: NEED TO USE THE BELOW UPDATE FUNCTIONS TO GET SOMETHING THAT TAKES STATE [0,1,1,2,...,0] AND TURNS IT INTO
    #   A (2,24,2) SHAPED INPUT INTO REPLY TABLE WHICH CAN BE DRAWN FROM IN TRAIN STEP
    def update(self, current_state, past_action, next_state, reward, next_turn):

        print("update replay table states: ", current_state, next_state)

        # UPDATE REPLAY TABLE AND TRAIN DQN MODEL
        self.DQN.record_state([current_state, past_action, next_state, reward, next_turn])
        self.DQN.train()


    def init_DQN(self):

        # INITIALISE THE DEEP Q LEARNING NETWORK
        input_shape = (4, 3, 3)
        output_size = self.env.num_actions
        self.DQN = DQN_model(input_shape, output_size, alpha=self.alpha, gamma=self.gamma)

    def store_model(self, i):

        # STORE MODEL WEIGHTS EVERY SO OFTEN. PATH TO SAVE MODEL WEIGHTS TO EVERY 5 GAMES (INCLUDED IN PATH NAME)
        checkpoint_path = "C:/Users/Emma/PycharmProjects/deep_q_learning/gym-dotsboxes/model_storage/cp-{" \
                          "epoch:04d}.ckpt "

        # Save the weights using the `checkpoint_path` format
        # TODO: THIS PROBABLY WON'T WORK UNTIL I FIX MODEL ARCH. NEEDS TO DO SAVE_WEGIHTS ON MODEL OBJECT
        self.DQN.save_weights(checkpoint_path.format(epoch=i))
