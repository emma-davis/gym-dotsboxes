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
        self.replay_table = None

        self.model = None
        self.build_DQN()

    def load_model(self, model_dir):
        # LOAD MODEL FROM GIVEN PATH
        self.saver.restore(self.sess, model_dir)

    def predict(self, state_matrix):
        # PREDICT Q VALUES USING MODEL
        q_values = self.model.predict(state_matrix, steps=10)  # TODO: ARBITRARY NUMBER OF STEPS HERE, IMPROVE THESE
        print("PREDICTED Q VALS: ", q_values)
        return q_values

    def train(self):
        # TRAIN CNN USING REPLAY TABLE AFTER CERTAIN AMOUNT OF TURNS PASSED
        if self.transition_count >= self.replay_size / 2:
            if self.transition_count == self.replay_size / 2:
                print("Replay Table Ready")

            random_tbl = np.random.choice(self.replay_table[:min(self.transition_count, self.replay_size)],
                                          size=self.update_size)

            print("Random Table: ", random_tbl)

            # GET INFO FROM REPLAY TABLE AND SPLIT INTO ITS PARTS
            feature_vectors = np.vstack(random_tbl['state'])
            print("VStack State: ", feature_vectors)
            actions = random_tbl['action']
            next_feature_vectors = np.vstack(random_tbl['next_state'])
            rewards = random_tbl['reward']
            next_turn_vector = random_tbl['had_next_turn']

            # GET NON TERMINAL STATE INDICES
            non_terminal_ix = np.where([~np.any(np.isnan(next_feature_vectors), axis=(1, 2, 3))])[1]
            next_turn_vector[next_turn_vector == 0] = -1

            q_current = self.predict(feature_vectors)

            # DEFAULT Q NEXT IS ALL ZEROS TO COVER TERMINAL STATES AND CREATE SHAPE
            q_next = np.zeros([self.update_size, self.output_size])
            q_next[non_terminal_ix] = self.predict(next_feature_vectors[non_terminal_ix])

            # TARGET SHOULD BE Q CURRENT. APPLY TANH TO REWARD TO FIT TO BETTER SCALE
            target = q_current.copy()
            rewards = np.tanh(rewards)

            # ONLY UPDATE ACTIONS TAKEN WITH REWARD, AND DO SO IN Q LEARNING WAY (BELLMAN EQU, SEE BELOW)
            # GRADIENT UPDATE ONLY APPLIED TO ACTION TAKEN TOO FOR GIVEN STATE
            target[np.arange(len(target)), actions] += (rewards + self.gamma * next_turn_vector * q_next.max(axis=1))

            # UPDATE MODEL
            self.sess.run(self.update_model, feed_dict={self.input_matrix: feature_vectors, self.target_Q: target})

    def build_DQN(self):

        # DEFINE CNN ARCHITECTURE

        # FINAL DECISION ON INPUT SHAPE:
        # (4, 3, 3) TUPLE: 3X3 IS BOXES AND 4 IS N/E/S/W LINE POSITION AROUND BOXES
        print("INPUT SHAPE: ", self.input_shape)
        print("OUTPUT SHAPE: ", self.output_size)
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
            optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'], loss_weights=None, weighted_metrics=None, run_eagerly=None
        )

        self.model = model
