"""
DEEP Q LEARNING NETWORK
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class DQN_model:

    def __init__(self, input_shape, output_size, alpha=0.001, gamma=0.9):
        self.replay_table_size = 20000
        self.update_size = 100
        self.input_shape = input_shape
        self.output_size = output_size
        self.alpha = alpha
        self.gamma = gamma

        self.session = None
        self.q_values = None
        self.input_matrix = None
        self.target_q = None
        self.update_model = None
        self.output_gradient = None

        self.transition_count = 0
        self.temp_transition_count = 0
        self.replay_table = None
        self.temp_replay_table = None
        self.replay_size = self.replay_table_size
        self.temp_replay_size = 20

        self.model = None
        self.build_DQN()

    def load_model(self, model_dir):
        # LOAD MODEL FROM GIVEN PATH
        self.saver.restore(self.sess, model_dir)

    def record_state(self, state):
        # RECORD STATE IN TEMP REPLAY TABLE, IF REWARD IS SET AS ANY VALUE THEN THIS IS THE LAST ACTION IN CURRENT
        # GAME, SO UPDATE THE MAIN REPLAY TABLE AND RESET TEMP TABLE BACK TO EMPTY FOR THE NEXT GAME
        check, _, _, reward, _ = state

        if check is not None:
            self.temp_replay_table[self.temp_transition_count % self.temp_replay_size] = state
            self.temp_transition_count += 1

        if reward in [-1, 0, 1]:
            # UPDATE REWARDS FOR ALL TURNS IN GAME USING GIVEN REWARD, THEN ADD TO MAIN REPLAY TABLE
            for i in range(self.temp_transition_count):
                if self.temp_replay_table[i][3] is None:
                    self.temp_replay_table[i][3] = reward
                elif self.temp_replay_table[i][3] > reward:
                    self.temp_replay_table[i][3] += reward
                else:
                    self.temp_replay_table[i][3] = reward
                self.replay_table[self.transition_count % self.replay_size] = self.temp_replay_table[i]
                self.transition_count += 1

            # FINALLY RESET TEMP TRANSITION COUNT AS 0 AND TEMP REPLAY TABLE AS EMPTY
            self.temp_transition_count = 0
            self.temp_replay_table = np.rec.array(np.zeros(self.temp_replay_size,
                                                           dtype=[('state', '(4,3,3)float32'),
                                                                  ('action', 'int8'),
                                                                  ('next_state', '(4,3,3)float32'),
                                                                  ('reward', 'float32'),
                                                                  ('next_mark', 'int8')]))

    def predict(self, state_matrix):
        # PREDICT Q VALUES USING MODEL
        q_values = self.model.predict(state_matrix)
        return q_values

    def train(self):
        # TRAIN CNN USING REPLAY TABLE AFTER CERTAIN AMOUNT OF TURNS PASSED
        if self.transition_count >= self.replay_size / 2:
            if self.transition_count == self.replay_size / 2:
                print("REPLAY TABLE READY TO TRAIN FROM...")

            random_tbl = np.random.choice(self.replay_table[:min(self.transition_count, self.replay_size)],
                                          size=self.update_size)

            # GET INFO FROM REPLAY TABLE AND SPLIT INTO ITS PARTS
            feature_vectors = np.vstack(random_tbl['state'])
            actions = random_tbl['action']
            next_feature_vectors = np.vstack(random_tbl['next_state'])
            rewards = random_tbl['reward']
            next_turn_vector = random_tbl['next_mark']

            # GET NON TERMINAL STATE INDICES
            # non_terminal_ix = np.where([~np.any(next_turn_vector is None)])[1]
            # next_turn_vector[next_turn_vector == 0] = -1

            q_current = self.predict(random_tbl['state'])

            # DEFAULT Q NEXT IS ALL ZEROS TO COVER TERMINAL STATES AND CREATE SHAPE
            # q_next = np.zeros([self.update_size, self.output_size])
            q_next = self.predict(random_tbl['next_state'])

            # TARGET SHOULD BE Q CURRENT. APPLY TANH TO REWARD TO FIT TO BETTER SCALE
            target = q_current.copy()
            rewards = np.tanh(rewards)

            # ONLY UPDATE ACTIONS TAKEN WITH REWARD, AND DO SO IN Q LEARNING WAY (BELLMAN EQU, SEE BELOW)
            # GRADIENT UPDATE ONLY APPLIED TO ACTION TAKEN TOO FOR GIVEN STATE
            target[np.arange(len(target)), actions] += (rewards + self.gamma * next_turn_vector * q_next.max(axis=1))

            # UPDATE MODEL
            #self.sess.run(self.update_model, feed_dict={self.input_matrix: feature_vectors, self.target_Q: target})
            self.model.fit(random_tbl['state'], target)

    def build_DQN(self):

        # DEFINE CNN ARCHITECTURE

        # FIRST, INITIALISE THE REPLAY TABLE
        self.replay_table = np.rec.array(np.zeros(self.replay_size,
                                                  dtype=[('state', '(4,3,3)float32'),
                                                         ('action', 'int8'),
                                                         ('next_state', '(4,3,3)float32'),
                                                         ('reward', 'float32'),
                                                         ('next_mark', 'int8')]))

        # ALSO INITIALISE THE TEMP REPLAY TABLE, WHICH TAKES ALL INFO FROM CURRENT GAME, UPDATES
        # REWARD AT THE END OF GAME FOR EACH LOT OF TURNS, UPDATES THE MAIN TABLE THEN RESETS TO TAKE NEXT GAME INFO
        self.temp_replay_table = np.rec.array(np.zeros(self.temp_replay_size,
                                                       dtype=[('state', '(4,3,3)float32'),
                                                              ('action', 'int8'),
                                                              ('next_state', '(4,3,3)float32'),
                                                              ('reward', 'float32'),
                                                              ('next_mark', 'int8')]))

        # FINAL DECISION ON INPUT SHAPE:
        # (4, 3, 3) TUPLE: 3X3 IS BOXES AND 4 IS N/E/S/W LINE POSITION AROUND BOXES
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

        # FLATTEN OUT TO FIT OUTPUT SHAPE, WHICH IS ONE PREDICTED Q VALUE FOR EACH OF 24 ACTIONS
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        output = model.add(layers.Dense(self.output_size))

        # GET SUMMARY OF MODEL ARCHITECTURE
        model.summary()

        # ESTABLISH OUTPUT Q_VALUES AND TARGET Q_VALUES TO CALCULATE LOSS
        self.q_values = output
        self.target_q = tf.keras.Input(shape=self.output_size, name='Target')

        model.compile(
            optimizer='sgd', loss=tf.keras.losses.MeanSquaredLogarithmicError(),
            metrics=['accuracy', tf.keras.metrics.MeanSquaredLogarithmicError()], loss_weights=None,
            weighted_metrics=None, run_eagerly=None
        )

        self.model = model
