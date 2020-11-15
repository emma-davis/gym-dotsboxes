from gym_dotsboxes.dotsboxes_env import *
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = 24#self.env.observation_space.shape
        #model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(24, input_dim=state_shape, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24))#model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            print("ACT RAND: ", self.env.sample())
            return self.env.sample()
        print(type(state[0]))
        print(list(state[0]))
        print("STATE: ", np.array(list(state[0])))
        print("ACT: ", np.argmax(self.model.predict(list(state[0]), batch_size=1)))#(np.array(list(state[0])))))
        return np.argmax(self.model.predict(state[0])[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    start_mark = random.choice(['A', 'B'])
    env = DotsBoxesEnv(start_mark) # NEED TO FIX HOW ACTIONS ARE DEDUCTED FROM AVAILABLE ONCE PLAYED (IN ENV CODE, NOT HERE)
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset()#.reshape(1, 2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            print(action)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            #new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# START GAME WHERE RANDOM AGENT PLAYS AGAINST Q LEARNING AGENT. A IS Q LEARNING AGENT, B IS RANDOM AGENT.
def random_vs_agent(max_episode=10):
    episode = 0
    start_mark = random.choice(['A', 'B'])
    agents = [Agent('A'), Agent('B')]
    env = DotsBoxesEnv(start_mark)

    # SET PARAMETERS
    alpha = 0.1  # THE LEARNING RATE
    gamma = 0.6  # DISCOUNT FACTOR
    epsilon = 0.1  # EXPLORATION VS EXPLOITATION FACTOR

    all_actions = env.available_actions()
    all_epochs = []

    # INITIALISE DIRECTIONAL GRAPH. THIS STORES THE AGENT'S DECISIONS, ACTS AS AN
    # ABRIDGED VERSION OF Q-VALUE MATRIX OF SORTS.
    g = nx.DiGraph()  # nx.read_graphml("training_5000.graphml")#

    while episode < max_episode:
        start_mark = random.choice(['A', 'B'])
        env.set_start_mark(start_mark)
        state = env.reset()

        _, mark, state_num = state
        epochs = 0
        done = False

        all_features_used = []
        all_feature_numbers = []
        all_actions_used = []
        mark = start_mark

        current_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        current_state_number = 0
        game_states = [0]

        # START PLAYING
        while not done:

            # MAKE SURE WE'RE ON THE CORRECT MARK
            mark = env.get_mark()
            env.print_turn(mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            unava_actions = list(set(all_actions) - set(ava_actions))

            actions_so_far = []

            if len(ava_actions) == 0:
                print("~~~~~ Finished: Draw ~~~~~")
                break

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # THIS WILL GENERATE ACTIONS BASED ON EXPLORATION VS EXPLOITATION FACTOR - HIGH EPSILON MEANS
            # MORE RANDOM ACTIONS. ELSE, PERFORM ACTION WHERE Q-VALUES HAVE BEEN LEARNED

            # IS AGENT IS B, PERFORM RANDOM ACTION FROM AVAILABLE ACTIONS. IF AGENT IS A, PERFORM ACTION
            # BASED ON Q LEARNING AND FEATURES.
            if mark == 'B':
                action = random.choice(ava_actions)
            else:
                action = 999
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(ava_actions)

                    # TODO: ADD FIX STOPPING THIS FROM HAPPENING!
                    if action in unava_actions:
                        print("Random action in unavailable actions!")

                    features = get_boolean_features(action, unava_actions)
                    features_num = int("".join(str(x) for x in features), 2)
                    all_actions_used.append(action)
                    all_features_used.append(features)
                    all_feature_numbers.append(features_num)

                    all_short_feature_numbers = [all_feature_numbers[i] for i in range(len(all_feature_numbers)) if
                                                 (all_feature_numbers[i] != 0) or all_feature_numbers[i] !=
                                                 all_feature_numbers[i - 1]]

                    # CHECK IF THIS ACTION EITHER COMPLETES OR SACRIFICES A BOX
                    if features[0] + features[1] > 0:
                        # THIS ACTION CLAIMS A SQUARE SO UPDATE GRAPH
                        update_edge(g, all_short_feature_numbers, 15)

                    if features[-1] + features[-2] > 0:
                        # THIS ACTION SACRIFICES A SQUARE SO UPDATE GRAPH
                        update_edge(g, all_short_feature_numbers, -15)

                else:
                    action = random.randrange(0, 23)
                    features = get_boolean_features(action, unava_actions)

                    # FIRST, CHECK IF CURRENT STATE EXISTS IN GRAPH. IF NOT THEN AGENT MUST ACT BLINDLY SO
                    # CHOOSE RANDOM
                    if g.has_node(tuple(game_states)) == False:
                        # CHOOSE RANDOM ACTION
                        print("CURRENT STATE NUMBER: ", tuple(game_states), " DOES NOT EXIST IN GRAPH.")
                        action = random.choice(ava_actions)
                        features = get_boolean_features(action, unava_actions)
                    else:
                        print("CURRENT STATE NUMBER: ", tuple(game_states), " DOES EXIST IN GRAPH.")
                        # GET FEATURES OF ALL AVAILABLE ACTIONS FROM THIS STATE
                        possible_features = []
                        for action in ava_actions:
                            possible_features.append(get_boolean_features(action, unava_actions))
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
                                if g.has_node(tuple(temp_game_states)) == True:
                                    print(tuple(game_states), tuple(temp_game_states))
                                    weight = g.get_edge_data(tuple(game_states), tuple(temp_game_states), default=None)[
                                        "weight"]
                                    print(weight)
                                    possible_weights.append(weight)
                                else:
                                    possible_weights.append(None)
                        print("Possible weights: ", possible_weights)

                        # FIND FEATURE THAT YIELDS HIGHEST WEIGHT AND GET THE STATE (MAYBE MULTIPLE) THIS
                        # CORRESPONDS TO. THERE MAY BE MULTIPLE OF THE SAME WEIGHT SO GET ALL MAX, NOT JUST FIRST
                        if possible_weights.count(None) == len(possible_weights):
                            # THERE ARE NO KNOWN WEIGHTS BETWEEN CURRENT STATE AND ANY OF THE POSSIBLE NEXT ONES,
                            # SO JUST GO FOR RANDOM ACTION.
                            action = random.choice(ava_actions)
                            features = get_boolean_features(action, unava_actions)

                        else:
                            # GET RID OF ALL None TO AVOID BREAKING MAX
                            possible_weights = [x for x in possible_weights if x is not None]
                            print("Possible weights: ", possible_weights)
                            m = max(possible_weights)
                            max_index = [i for i, j in enumerate(possible_weights) if j == m]

                            # IF MAX POSSIBLE WEIGHTS IS LESS THAN 0, THEN JUST PICK RANDOM ACTION
                            if m < 0:
                                action = random.choice(ava_actions)
                                features = get_boolean_features(action, unava_actions)
                            else:
                                features = possible_features[random.choice(max_index)]

                                # GET ACTION THAT WILL LEAD TO THIS. THERE MAY BE MANY ACTIONS, SO PICK RANDOM FROM THEM.
                                possible_actions = []
                                for action in ava_actions:
                                    if get_boolean_features(action, unava_actions) == features:
                                        possible_actions.append(action)

                                action = random.choice(possible_actions)


                    # UPDATE GAME_STATES VECTOR WITH NEW STATE TRAVELLED TO, MAKING SURE TO NOT HAVE 0 MULTIPLE TIMES IN ROW
                    if features_num == 0 & game_states[-1] == 0:
                        print("No changes to game_states: ", game_states)
                    else:
                        game_states.append(features_num)
                        print("Game states: ", game_states)

            print("ACTION: " + str(action))


            next_state, reward, done, info = env.step(action)

            # NEWLY ADDED BELOW
            _, next_mark, next_state_num = next_state

            env.render()

            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark, state_num = state

            # CALCULATE NEW Q-VALUE FOR Q(STATE, ACTION) JUST PERFORMED AND REPLACE OLD VALUE
            # new_value = ((1 - alpha) * old_value) + (alpha * (reward + gamma * next_max))

            state = next_state
            epochs += 1

            # action = agent.act(state, ava_actions)
            # state, reward, done, info = env.step(action)
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        # SWITCH STARTING AGENT FOR EACH GAME
        # start_mark = next_mark(start_mark)
        episode += 1
        print(all_features_used)
        print(all_actions_used)

        # UPDATE FEATURE WEIGHTS BASED ON PREVIOUS GAME
        result = env.return_result()

        if result == 'A':
            # IF OUR Q LEARNING AGENT A HAS WON, THEN LOOK AT FEATURES USED AND REWARD
            features_sum = np.sum(all_features_used, axis=0)
            print(features_sum)


        elif result == 'B':
            # IF OUR Q LEARNING AGENT A HAS LOST, THEN LOOK AT FEATURES USED AND REWARD
            features_sum = np.sum(all_features_used, axis=0)
            print(features_sum)



        elif result == 'DRAW':
            # IF OUR Q LEARNING AGENT A HAS DRAWN, THEN WE STILL WANT TO PENALISE SLIGHTLY
            features_sum = np.sum(all_features_used, axis=0)
            print(features_sum)


        else:
            error_count += 1
            print("Something has gone wrong, game has ended with no clear winner or draw.")
"""