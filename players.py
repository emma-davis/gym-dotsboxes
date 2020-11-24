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
import math
import networkx as nx


class UserPlayer:

    # LETS USER PLAY GAME BY ENTERING THEIR ACTIONS AS NUMBERS
    def __init__(self, env):
        self.env = env

    # def set_environment(self, environment):
    #    self.env = environment

    def act(self, state):

        if self.env is None:
            raise ValueError("Environment has not been set.")

        # USER INPUTS CHOSEN VALID ACTION BY ENTERING NUMBER 0-23
        action = None
        while action not in self.env.available_actions:
            print("Valid actions: ", self.env.available_actions)
            action = int(input("Please choose an action:\n"))

        # APPLY ACTION TO ENVIRONMENT VIA STEP FUNCTION DEFINED IN
        # ENVIRONMENT CLASS
        # self.env.step(action)

        return action


class ApproxFeaturePlayer:

    # PREVIOUS Q LEARNING MODEL THAT USED LINEAR APPROXIMATION OF FEATURES. IS GREEDY IN NATURE
    # DUE TO BIAS IN FEATURE SELECTION AND INABILITY TO LEARN COMPLEX STRATEGY
    def __init__(self, env):
        self.env = env
        self.g = nx.read_graphml("15000_q_v1.6_vs_smarter_added_complete_square_force.graphml")

    # CREATE LIST OF BOOLS REPRESENTING FEATURES OF STATE-ACTION PAIR FED THROUGH
    def get_boolean_features(self, action, unava_actions):
        # ORDER: 'completes_square', 'completes_two_squares', 'creates_small_chain', 'creates_medium_chain',
        # 'creates_large_chain', 'creates_small_circle', 'creates_medium_circle', 'creates_large_circle',
        # 'sacrifices_square', 'sacrifices_two_squares'
        #
        # MISSING: 'creates_small_cycle', 'creates_medium_cycle', 'creates_large_cycle'
        features = []
        all_actions = unava_actions + [action]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1 & 2: COMPLETES SQUARE? COMPLETES 2 SQUARES?
        square_combos = [[0, 3, 4, 7], [1, 4, 5, 8], [2, 5, 6, 9], [7, 10, 11, 14], [8, 11, 12, 15], [9, 12, 13, 16],
                         [14, 17, 18, 21], [15, 18, 19, 22], [16, 19, 20, 23]]
        counter = 0

        # FOR EVERY SQUARE WHERE THREE OF THE SIDES ARE IN UNAVA ACTIONS AND ONE IS THE ACTION, ADD TO COUNTER (0 <= counter <= 2)
        for square in square_combos:
            if (action in square) & (len(list(set(unava_actions) & set(square))) == 3):
                counter += 1

        if counter == 0:
            features.append(0)
            features.append(0)
        elif counter == 1:
            features.append(1)
            features.append(0)
        elif counter == 2:
            features.append(1)
            features.append(1)
        else:
            print("This shouldn't happen!")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3, 4 & 5: COMPLETES SMALL/MEDIUM/LARGE CHAIN?
        # Note: A chain is an open ended string of squares which starts with a square with 3 sides completed. An action is said to
        # create a chain when it is the action that completes the 3 sides complete square at the start of the chain.
        # small chain = 2   medium chain = 3-4  large chain = 5-9

        # FIRST, CHECK IF THIS ACTION FORMS THE THIRD SIDE OF ONE (OR TWO) SQUARES WITHOUT FORMING THE FOURTH SIDE OF ANY SQUARES.
        # THIS COUNTER SHOULD ALSO BE USED FOR 'sacrifices_square' AND 'sacrifices_two_squares' FEATURES

        # NOTE: THIS ONLY WORKS FOR sacrifice_counter == 1, AS TWO CHAINS CREATED AT SAME TIME IS POSSIBLE BUT NOT CONSIDERED IN THIS
        sacrifice_counter = 0
        incomplete_side = 999
        start_square = []

        for square in square_combos:
            if (action in square) & (len(list(set(all_actions) & set(square))) == 3):
                sacrifice_counter += 1
                incomplete_side = list(set(square) - set(all_actions))[0]
                start_square = square

        # SECOND, CHECK HOW LONG THE CHAIN IS. THIS IS DONE BY LOOKING AT THE SQUARE THAT IS ATTACHED TO SEMI-COMPLETE ONE VIA THE
        # INCOMPLETE SIDE, THEN REPEAT FOR THAT SQUARE KEEPING TABS ON HOW MANY SQUARES
        squares_in_chain = 1

        if incomplete_side != 999:
            # NEED TO FIND THE SQUARE THAT INCLUDES INCOMPLETE_SIDE AND ALSO HAS ONLY TWO SIDES COMPLETE (WHERE INCOMPLETE SIDE IS NOT ONE OF THEM)
            temp_all_actions = all_actions
            while squares_in_chain < 9:
                no_chain = True
                for square in square_combos:
                    if (incomplete_side in square) & (square != start_square):
                        if len(list(set(square) - set(all_actions))) == 2:
                            # ADD SQUARE TO THE CHAIN TALLY, THEN PRETEND THE PREVIOUS INCOMPLETE SIDE IS NOW COMPLETE, ALLOWING US TO FIND THE NEXT
                            # INCOMPLETE SIDE OF NEXT SQUARE
                            squares_in_chain += 1
                            temp_all_actions.append(incomplete_side)
                            incomplete_side = list(set(square) - set(temp_all_actions))[0]
                            no_chain = False
                # IF THERE ARE NO MORE SQUARES FITTING THIS CRITERIA THEN CHAIN HAS ENDED, BREAK WHILE LOOP
                if no_chain == True:
                    break

            # NOW LOOK AT SQUARES_IN_CHAIN TALLY AND ALLOCATE FEATURES ACCORING TO CRITERIA IN COMMENT ABOVE
            if squares_in_chain == 2:
                features.append(1)
                features.append(0)
                features.append(0)
            elif squares_in_chain in [3, 4]:
                features.append(0)
                features.append(1)
                features.append(0)
            elif squares_in_chain in [5, 6, 7, 8, 9]:
                features.append(0)
                features.append(0)
                features.append(1)
            else:
                features.append(0)
                features.append(0)
                features.append(0)
        else:
            features.append(0)
            features.append(0)
            features.append(0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6, 7 & 8: COMPLETES SMALL/MEDIUM/LARGE CIRCLE?

        # Note: A circle is a group of squares made up of completed edges with no completed edges inside. Small cicles are 2x2 squares,
        # medium circles are 2x3 or 3x2 squares, and large circles are 3x3 squares (edge around whole board with nothing else inside it).
        small_circle_combos = [[0, 1, 3, 5, 10, 12, 14, 15], [1, 2, 4, 6, 11, 13, 15, 16],
                               [7, 8, 10, 12, 17, 19, 21, 22], [8, 9, 11, 13, 18, 20, 22, 23]]
        medium_circle_combos = [[0, 1, 2, 3, 6, 10, 13, 14, 15, 16], [7, 8, 9, 10, 13, 17, 20, 21, 22, 23],
                                [0, 1, 3, 5, 10, 12, 17, 19, 21, 22], [1, 2, 4, 6, 11, 13, 18, 20, 22, 23]]
        large_circle_combo = [0, 1, 2, 3, 6, 10, 13, 17, 20, 21, 22, 23]

        small_not_allowed_pattern = [4, 7, 8,
                                     11]  # THESE ARE THE EDGES INSIDE SMALL CIRCLE. DO PATTERN + START EDGE NUMBER
        medium_not_allowed_horizontal_pattern = [4, 5, 7, 8, 9, 11, 12]
        medium_not_allowed_vertical_pattern = [4, 7, 8, 11, 14, 15, 18]

        # IF UNAVA ACTIONS + ACTION AND ANY OF THE ABOVE COMBOS ARE EXACTLY THE SAME WITH NO EXTRA/MISSING ACTIONS, THEN THERE IS A CIRCLE
        counter = 0
        for circle in small_circle_combos:
            start_edge = circle[0]
            not_allowed_edges = [x + start_edge for x in small_not_allowed_pattern]
            # FIRST, CHECK THERE ARE NO EDGES INSIDE THE CIRCLE, AS THIS STOPS IT BEING A CIRCLE
            if list(set(not_allowed_edges) & set(all_actions)) == []:
                # NEXT, CHECK THAT ALL CIRCLE EDGES HAVE BEEN PLACED, WITH THE LAST EDGE BEING PLACED IN THIS TURN.
                if (list(set(all_actions) & set(circle)) == circle) & (action in circle):
                    counter = 1
                    features.append(1)
                    features.append(0)
                    features.append(0)
        for circle in medium_circle_combos:
            start_edge = circle[0]
            not_allowed_edges = []
            if [1, 22] in circle:
                not_allowed_edges = [x + start_edge for x in medium_not_allowed_vertical_pattern]
            else:
                not_allowed_edges = [x + start_edge for x in medium_not_allowed_horizontal_pattern]
            # FIRST, CHECK THERE ARE NO EDGES INSIDE THE CIRCLE, AS THIS STOPS IT BEING A CIRCLE
            if list(set(not_allowed_edges) & set(all_actions)) == []:
                # NEXT, CHECK THAT ALL CIRCLE EDGES HAVE BEEN PLACED, WITH THE LAST EDGE BEING PLACED IN THIS TURN.
                if (list(set(all_actions) & set(circle)) == circle) & (action in circle):
                    counter = 2
                    features.append(0)
                    features.append(1)
                    features.append(0)
        if set(all_actions) == set(large_circle_combo):
            counter = 3
            features.append(0)
            features.append(0)
            features.append(1)

        if counter == 0:
            features.append(0)
            features.append(0)
            features.append(0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 12 & 13: SACRIFICES SQUARE? SACRIFICES TWO SQUARES?
        if sacrifice_counter == 0:
            features.append(0)
            features.append(0)
        elif sacrifice_counter == 1:
            features.append(1)
            features.append(0)
        elif sacrifice_counter == 2:
            features.append(1)
            features.append(1)
        else:
            print("This shouldn't happen!")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return features

    def act(self, current_state):
        action = random.randrange(0, 23)
        ava_actions = self.env.available_actions
        unava_actions = np.subtract([i for i in range(self.num_actions)], ava_actions)
        features = self.get_boolean_features(action, unava_actions)
        game_states = [0]

        # FIRST, CHECK IF CURRENT STATE EXISTS IN GRAPH. IF NOT THEN AGENT MUST ACT BLINDLY SO
        # CHOOSE RANDOM
        if g.has_node(tuple(game_states)) == False:
            # CHOOSE RANDOM ACTION
            print("CURRENT STATE NUMBER: ", tuple(game_states), " DOES NOT EXIST IN GRAPH.")
            action = random.choice(ava_actions)
            features = self.get_boolean_features(action, unava_actions)
        else:
            print("CURRENT STATE NUMBER: ", tuple(game_states), " DOES EXIST IN GRAPH.")
            # GET FEATURES OF ALL AVAILABLE ACTIONS FROM THIS STATE
            possible_features = []
            for action in ava_actions:
                possible_features.append(self.get_boolean_features(action, unava_actions))
            possible_features = [list(i) for i in
                                 set(tuple(i) for i in possible_features)]  # list(set(possible_features))

            # GET BINARY NUMBERS FROM FEATURES, AS THIS IS HOW WE REPRESENT THEM IN GRAPH
            possible_features_numbers = []

            for feature in possible_features:
                possible_features_numbers.append(int("".join(str(x) for x in feature), 2))

            print("Possible features: ", possible_features)
            print("Possible features numbers: ", possible_features_numbers)

            # GET WEIGHT OF EDGE BETWEEN CURRENT STATE (FEATURES) AND ALL POSSIBLE NEXT STATES.
            # THESE WEIGHTS MAY OR MAY NOT EXIST. IN THE CASE WHERE THEY DO NOT EXIST, None.
            possible_weights = []
            for features_number in possible_features_numbers:
                # FIRST CHECK STATE NODE EXISTS, THEN IF THERE IS A WEIGHT ON EDGE BETWEEN NODES
                print("FEATURES NUMER: ", features_number)
                print("GAME STATES: ", game_states)
                if features_number == 0 & game_states[-1] == 0:
                    print("0 TO 0 SO NO WEIGHTS HERE.")
                    possible_weights.append(None)
                else:
                    temp_game_states = list(game_states)
                    temp_game_states.append(features_number)
                    print(temp_game_states)
                    if g.has_node(tuple(temp_game_states)):
                        print(tuple(game_states), tuple(temp_game_states))
                        weight = g.get_edge_data(tuple(game_states), tuple(temp_game_states), default=None)["weight"]
                        print(weight)
                        possible_weights.append(weight)
                    else:
                        possible_weights.append(None)
            print("Possible weights: ", possible_weights)

            # FIND FEATURE THAT YIELDS HIGHEST WEIGHT AND GET THE STATE (MAYBE MULTIPLE) THIS
            # CORRESPONDS TO. THERE MAY BE MULTIPLE OF THE SAME WEIGHT SO GET ALL MAX, NOT JUST FIRST
            if possible_weights.count(None) == len(possible_weights):
                print("NO WEIGHTS, RANDOM ACTION.")
                # THERE ARE NO KNOWN WEIGHTS BETWEEN CURRENT STATE AND ANY OF THE POSSIBLE NEXT ONES,
                # SO JUST GO FOR RANDOM ACTION.
                action = random.choice(ava_actions)
                features = self.get_boolean_features(action, unava_actions)

            else:
                # GET RID OF ALL None TO AVOID BREAKING MAX
                possible_weights_no_none = [x for x in possible_weights if x is not None]
                print("Possible weights: ", possible_weights_no_none)
                m = max(possible_weights_no_none)
                max_index = [i for i, j in enumerate(possible_weights) if j == m]

                # IF MAX POSSIBLE WEIGHTS IS LESS THAN 0, THEN JUST PICK RANDOM ACTION
                if m < 0:
                    print("MAX POSSIBLE WEIGHT IS LESS THAN 0.")

                    # GET ALL ACTIONS THAT DON'T DON'T LEAD TO STATES WITH FEATURES THAT PRODUCE NEGATIVE WEIGHTS
                    # GET LIST OF ALL FEATURES THAT LEAD TO NEGATIVE OUTCOMES
                    negative_features_index = []  # [i for i, j in enumerate(possible_weights) if (j < 0 & j != None)]
                    for i in range(len(possible_weights)):
                        if possible_weights[i] != None:
                            if possible_weights[i] < 0:
                                negative_features_index.append(i)
                    print("INDEX OF NEGATIVE FEATURES ARE: ", negative_features_index)
                    negative_features = []  # [possible_features for i in negative_features_index]
                    for index in negative_features_index:
                        negative_features.append(possible_features[index])
                    print("NEGATIVE FEATURES ARE: ", negative_features)

                    # GET LIST OF ACTIONS THAT WE CAN PICK FROM (I.E. ACTIONS THAT DON'T LEAD TO ANY STATES WITH FEATURES IN LIST)
                    actions_pool = []
                    for i in ava_actions:
                        features = self.get_boolean_features(i, unava_actions)
                        if features not in negative_features:
                            actions_pool.append(i)

                    if len(actions_pool) == 0:
                        # THERE ARE NO ACTIONS THAT DON'T LEAD TO A STATE WITH A NEGATIVE WEIGHT SO JUST PICK ANY RANDOM ONE.
                        print("NO ACTIONS WITH POSSIBILITY OF +VE REWARD.")
                        action = random.choice(ava_actions)
                    else:
                        # THERE ARE ACTIONS THAT DON'T LEAD TO A STATE WITH A NEGATIVE WEIGHT, PICK FROM THIS POOL
                        print("ACTIONS WITH POSSIBILITY OF +VE REWARD EXIST - PICKING FROM THIS POOL...")
                        print(actions_pool)
                        action = random.choice(actions_pool)
                    print("ACTION: ", action)
                    features = self.get_boolean_features(action, unava_actions)
                else:
                    print("MAX POSSIBLE WEIGHT NOT LESS THAN 0.")
                    features = possible_features[random.choice(max_index)]
                    print("CHOSEN FEATURES TO FIND ACTION LEADING TO ARE: ", features)

                    # GET ACTION THAT WILL LEAD TO THIS. THERE MAY BE MANY ACTIONS, SO PICK RANDOM FROM THEM.
                    possible_actions = []
                    for action in ava_actions:
                        if self.get_boolean_features(action, unava_actions) == features:
                            possible_actions.append(action)

                    action = random.choice(possible_actions)
                    print("ACTION: ", action)

        return action


class GreedyPlayer:

    # GREEDY PLAYER ALWAYS COMPLETES A BOX IF IT CAN
    def __init__(self, env):
        self.env = env

    def act(self, current_state, i):

        # USING BOARD CURRENT STATE, CHOOSE AN OPTION THAT EITHER
        # - COMPLETES A BOX
        # - FAILING THIS, RANDOM ACTION
        if self.env is None:
            raise ValueError("Environment has not been set.")

        # INIT DECISION LISTS
        greedy_actions = []
        square_combos = [[0, 3, 4, 7], [1, 4, 5, 8], [2, 5, 6, 9],
                         [7, 10, 11, 14], [8, 11, 12, 15], [9, 12, 13, 16],
                         [14, 17, 18, 21], [15, 18, 19, 22], [16, 19, 20, 23]]

        for action in self.env.available_actions:
            # ITERATE THROUGH ALL SQUARE COMBINATIONS TO SEE IF THIS CURRENT ACTION
            # IS THE ONE THAT WILL COMPLETE SQUARE(S), THUS GAINING AGENT REWARDS
            for square in square_combos:
                temp = square.copy()
                if action in square:
                    temp.remove(action)
                    occupied_square_count = 0
                    for x in temp:
                        if x not in self.env.available_actions:
                            occupied_square_count += 1

                    # WANT THE SCORE TO BE 3 NOT 4, AS AT THIS POINT THE ACTION
                    # HASN'T HAPPENED, SO THE SQUARE SHOULD ONLY HAVE 3 SIDES
                    # AND NOT 4
                    if occupied_square_count == 3:
                        greedy_actions.append(action)
                    if occupied_square_count > 3:
                        print("~~~~~This shouldn't happen!~~~~~")

        if len(greedy_actions) == 0:
            action = random.choice(self.env.available_actions)
        else:
            action = random.choice(greedy_actions)

        # APPLY ACTION TO ENVIRONMENT VIA STEP FUNCTION DEFINED IN
        # ENVIRONMENT CLASS
        # self.env.step(action)

        # MAKE AGENT RANDOM AT START THEN SLOWLY MOVE TO FULLY GREEDY FROM THERE
        if i < 10000:
            action = random.choice(self.env.available_actions)
        elif i < 25000:
            x = random.randint(0, 1)
            if x == 1:
                action = random.choice(self.env.available_actions)

        return action


class DQNPlayer:

    def __init__(self, env, alpha=0.01, epsilon=0.05, gamma=0.85):
        self.env = env
        self.learning = True
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.DQN = None

        self.past_state = None
        self.past_action = None

    def act(self, current_state):

        # CHOOSES ACTION BASED ON DQN MODEL HIGHEST Q-VALUE PREDICTED
        # current_state = self.env.state
        matrix = self.get_3D_matrix(current_state)

        # IF RANDOM NUMBER LESS THAN EPSILON, CHOOSE RANDOM ACTION, OTHERWISE
        # PICK VALID ACTION THAT YIELDS HIGHEST PREDICTED Q VALUE FROM DQN MODEL
        if random.random() < self.epsilon:
            action = np.random.choice(self.env.available_actions)
        else:
            q_values = self.DQN.predict(matrix)[0]
            print(q_values)
            max_q = q_values[self.env.available_actions].max()
            best_actions = np.where(q_values == max_q)[0]

            if len([action for action in best_actions if action in self.env.available_actions]) is not 0:
                print("~~~~~~~~~~~CHOOSING VIA DQN...~~~~~~~~~~~~~")
                action = random.choice([action for action in best_actions if action in self.env.available_actions])
            else:
                action = np.random.choice(self.env.available_actions)

        self.past_state = matrix.copy()
        self.past_action = action

        return action

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
        return matrix

    def get_training_info(self, state, reward):
        # COULD PROBABLY JUST DO THIS IN TRAIN_MODEL
        have_next_turn = int(self._environment.current_player == self)

        if self._environment.state is not None:
            next_feature_vector = self.generate_input_vector(state)
        else:
            next_feature_vector = None

        self.update(self.last_state, self.last_action, next_feature_vector, reward, have_next_turn)

    def update(self, current_state, past_action, next_state, reward, next_turn):

        # UPDATE REPLAY TABLE AND TRAIN DQN MODEL
        self.DQN.record_state((current_state, past_action, next_state, reward, next_turn))
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

        # SAVE MODEL WEIGHTS AS CHECKPOINT
        self.DQN.save_weights(checkpoint_path.format(epoch=i))
