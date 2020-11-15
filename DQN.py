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
        #self.loss_func = None
        #self.optimizer = None
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
        #q_values = self.sess.run(self.q_values, feed_dict={self.input_matrix:state_matrix})
        q_values = self.model.predict(state_matrix, steps=10) # ARBITRARY NUMBER OF STEPS HERE
        print("PREDICTED Q VALS: ", q_values)
        return q_values

    def train(self):
        """
        Train the network based on replay table information
        """
        if self.transition_count >= self.replay_size / 2:
            if self.transition_count == self.replay_size / 2:
                print("Replay Table Ready")

            random_tbl = np.random.choice(self.replay_table[:min(self.transition_count, self.replay_size)],
                                          size=self.update_size)

            print("Random Table: ", random_tbl)

            # Get the information from the replay table
            # TODO: NEED TO MAKE SURE WHOLE TRAINING FUNCTION WORKS WITH
            feature_vectors = np.vstack(random_tbl['state'])
            print("VStack State: ", feature_vectors)
            actions = random_tbl['action']
            next_feature_vectors = np.vstack(random_tbl['next_state'])
            rewards = random_tbl['reward']
            next_turn_vector = random_tbl['had_next_turn']

            # Get the indices of the non-terminal states
            non_terminal_ix = np.where([~np.any(np.isnan(next_feature_vectors), axis=(1, 2, 3))])[1]
            next_turn_vector[next_turn_vector == 0] = -1

            q_current = self.predict(feature_vectors)
            # Default q_next will be all zeros (this encompasses terminal states)
            q_next = np.zeros([self.update_size, self.output_size])
            q_next[non_terminal_ix] = self.predict(next_feature_vectors[non_terminal_ix])

            # The target should be equal to q_current in every place
            target = q_current.copy()

            # Apply hyperbolix tangent non-linearity to reward
            rewards = np.tanh(rewards)

            # Only actions that have been taken should be updated with the reward
            # This means that the target - q_current will be [0 0 0 0 0 0 x 0 0....]
            # so the gradient update will only be applied to the action taken
            # for a given feature vector.
            # The next turn vector controls for a conditional minimax. If the opponents turn is next,
            # The value of the next state is actually the negative maximum across all actions. If our turn is next,
            # The value is the maximum.
            target[np.arange(len(target)), actions] += (rewards + self.gamma * next_turn_vector * q_next.max(axis=1))

            # Update the model
            self.sess.run(self.update_model, feed_dict={self.input_matrix: feature_vectors, self.target_Q: target})

    def build_DQN(self):

        # DEFINE CNN ARCHITECTURE WITH 2 CONV LAYERS

        # FINAL DECISION ON INPUT SHAPE:
        # (3, 3, 4) TUPLE: 3X3 IS BOXES AND 4 IS N/E/S/W LINE POSITION AROUND BOXES. POSSIBLY (4,3,3) INSTEAD?
        print("INPUT SHAPE: ", self.input_shape)
        print("OUTPUT SHAPE: ", self.output_size)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

        # FLATTEN OUT TO FIT OUTPUT SHAPE, WHICH IS ONE PREDICTED Q VALUE FOR EACH OF 24 ACTIONS
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        output = model.add(layers.Dense(self.output_size))

        # GET SUMMARY OF MODEL ARCHITECTURE
        model.summary()

        """
        model = tf.keras.models.Sequential([
            # FIRST CONVOLUTION
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),

            # SECOND CONVOLUTION
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            # FLATTEN TO FEED INTO DENSE NN
            tf.keras.layers.Flatten(),

            # 256 NEURON DENSE LAYER
            tf.keras.layers.Dense(256, activation='relu'),

            # OUTPUT LAYER WITH 24 POSSIBILITIES, ONE NODE FOR EACH ACTION
            tf.keras.layers.Dense(24, activation='relu')
        ])
        """

        # ESTABLISH OUTPUT Q_VALUES AND TARGET Q_VALUES TO CALCULATE LOSS
        self.q_values = output
        self.target_q = tf.keras.Input(shape=self.output_size, name='Target')

        # TODO: CANT USE Q_VALUES AND TARGET_Q HERE YET? MAKE SURE THE OPTIMIZER AND LOSS FUNCTIONS FIT WHAT TRYING
        #  TO ACHEIVE
        model.compile(
            optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(),#tf.keras.losses.mean_squared_error(self.target_q, self.q_values),
            metrics=['accuracy'], loss_weights=None, weighted_metrics=None, run_eagerly=None
        )

        self.model = model