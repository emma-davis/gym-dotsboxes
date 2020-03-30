# IMPORTS
import gym
import numpy as np
import random
import time

# CREATE ENVIRONMENT
env = gym.make('Taxi-v3')

# ~~~ METHOD 2: Q LEARNING ~~~
# Q-values mapped to (state, action) help agent learn from rewards over time. Q-values updated by equation:
#   Q(state, action) <- (1 - a)Q(state, action) + a(reward + ymaxQ(next state, all actions))
# a is the learning rate - the extent to which q-values updated every iteration)
# y is the discount factor - determines importance we want to give future rewards

# CREATES EMPTY Q-VALUES MATRIX THE SIZE OF OBSERVATION X ACTION SPACE
q_values = np.zeros([env.observation_space.n, env.action_space.n])

# START TRAINING TIMER
start = time.perf_counter()

# SET PARAMETERS
alpha = 0.1  # THE LEARNING RATE
gamma = 0.6  # DISCOUNT FACTOR
epsilon = 0.1  # EXPLORATION VS EXPLOITATION FACTOR

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    #env.render()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        # THIS WILL GENERATE ACTIONS BASED ON EXPLORATION VS EXPLOITATION FACTOR - HIGH EPSILON MEANS
        # MORE RANDOM ACTIONS. ELSE, PERFORM ACTION WHERE Q-VALUES HAVE BEEN LEARNED
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            print(state)
            # GETS POSITION OF MAX REWARD ALONG X AXIS OF STATE IN Q-VALUE MATRIX
            action = np.argmax(q_values[state])

        next_state, reward, done, info = env.step(action)

        # STORE OLD Q-VALUE AS VAR TO USE IN UPDATE STEP, GET VALUE OF MAX REWARD NEXT STATE
        # FROM Q-VALUE MATRIX TO ALSO USE IN UPDATE STEP
        old_value = q_values[state, action]
        next_max = np.max(q_values[next_state])

        # CALCULATE NEW Q-VALUE FOR Q(STATE, ACTION) JUST PERFORMED AND REPLACE OLD VALUE
        new_value = ((1 - alpha) * old_value) + (alpha * (reward + gamma * next_max))
        q_values[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    #if i % 100 == 0:
        #print("Episode: ", i)

end = time.perf_counter()
print("Training finished. Time taken: ", end - start)

# ~~~ EVALUATION OF AGENT ~~~
total_epochs, total_penalties = 0, 0
episodes = 100

for i in range(1):#range(episodes):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        env.render()
        # ACTION IS BASED ON THE Q-VALUES MATRIX AGENT HAS LEARNED - IT IS PICKING WHAT IT HAS LEARNED TO BE THE
        # BEST MOVE IN ITS CURRENT STATE
        action = np.argmax(q_values[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

# CHECK IF EVAL METHOD WORKS, MORE TIMESTEPS PER EP THAN EXPECTED
print("Results after ", episodes, " episodes:\nAverage timesteps per episode: ", total_epochs,
      "\nAverage penalties per episode: ", total_penalties)
