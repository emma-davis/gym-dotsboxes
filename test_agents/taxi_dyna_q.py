# IMPORTS
import gym
import numpy as np
import random
import time

# CREATE GYM ENVIRONMENT
env = gym.make('Taxi-v3')


# FUNCTION THAT UPDATES Q VALUES MATRIX WITH INFORMATION LEARNED FROM PERFORMING ACTION
def q_learning(q_values, action, state, next_state, alpha, gamma):
    old_value = q_values[state, action]
    next_max = np.max(q_values[next_state])

    # CALCULATE NEW Q-VALUE FOR Q(STATE, ACTION) JUST PERFORMED AND REPLACE OLD VALUE USING
    # SIMPLE VALUE ITERATION UPDATE
    new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
    q_values[state, action] = new_value

    return q_values


# FUNCTION THAT PERFORMS Q PLANNING USING RANDOM MODEL SIMULATIONS OF (STATE, ACTION) PAIRS TO UPDATE Q VALUES
def q_planning(model, q_values, episode_time, time_weight):
    # RANDOMLY CHOOSE STATE FROM MODEL, THEN RANDOMLY CHOOSE ACTION TO TAKE FROM THAT STATE
    rand_s_index = np.random.choice(range(len(model.keys())))
    rand_state = list(model)[rand_s_index]

    rand_a_index = np.random.choice(range(len(model[rand_state].keys())))
    rand_action = list(model[rand_state])[rand_a_index]

    # GET INFO FROM MODEL ABOUT REWARD, TIME AND NEXT STATE EXPECTED FROM (STATE, ACTION) PAIR
    reward, next_state, time = model[rand_state][rand_action]

    # UPDATE REWARD BASED ON TIME (THIS IS THE DYNA Q+ ELEMENT)
    reward += time_weight * np.sqrt(episode_time - time)

    old_value = q_values[state, action]
    next_max = np.max(q_values[next_state])

    # CALCULATE NEW Q-VALUE FOR Q(STATE, ACTION) JUST PERFORMED AND REPLACE OLD VALUE USING
    # SIMPLE VALUE ITERATION UPDATE
    new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
    q_values[state, action] = new_value

    return q_values


# FUNCTION THAT UPDATES THE MODEL
def update_model(model, action, reward, state, next_state, episode_time):
    if state not in model.keys():
        model[state] = {}
    # TODO: INSERT SOMETHING HERE PENALISING ACTIONS THAT CANNOT BE TAKEN & SETTING THEIR NEXT_STATE=STATE?
    model[state][action] = (reward, next_state, episode_time)
    return model


# SET MODEL PARAMETERS
alpha = 0.1  # THE LEARNING RATE
gamma = 0.6  # DISCOUNT FACTOR
epsilon = 0.1  # EXPLORATION VS EXPLOITATION FACTOR
episode_time = 0  # FOR DYNA Q+ TIME BASED PLANNING
time_weight = 0.0001  # ARBITRARY WEIGHTING FOR TIME ELEMENT OF REWARD #TODO: LOOK INTO THIS DYNA Q+ BIT
steps = 5  # ARBITRARY NUMBER OF TIMES MODEL SHOULD BE UPDATED IN Q-PLANNING
episodes = 10000  # NUMBER OF EPISODES TO RUN IN TRAINING (THESE ARE FULL RUNS UNTIL OBJECTIVE IS ACHIEVED)

total_epochs = []
total_penalties = []

# CREATE EMPTY Q-VALUES MATRIX OF CORRECT SIZE AND EMPTY DICT FOR MODEL
q_values = np.zeros([env.observation_space.n, env.action_space.n])
model = {}

# START TRAINING TIMER
start = time.perf_counter()

# BEGIN TRAINING MODEL
for i in range(episodes):

    # RESET EVERYTHING
    state = env.reset()
    epochs, penalties, reward, episode_time = 0, 0, 0, 0
    done = False

    # STARTS NEW EPISODE OF LEARNING
    while not done:

        # GENERATE ACTIONS BASED ON EPSILON (EXPLORATION VS EXPLOITATION)
        if random.uniform(0, 1) < epsilon:

            # PERFORM RANDOM ACTION FROM CURRENT STATE
            action = env.action_space.sample()
        else:

            # GET ACTION THAT YIELDS HIGHEST REWARD FROM CURRENT STATE
            action = np.argmax(q_values[state])

        # GETS INFO ABOUT PERFORMING CHOSEN ACTION IN ENVIRONMENT
        next_state, reward, done, info = env.step(action)

        # UPDATE Q VALUES MATRIX IN Q LEARNING STEP, USING INFORMATION GAINED ABOUT PREVIOUS
        # ACTION IN ENVIRONMENT
        q_values = q_learning(q_values, action, state, next_state, alpha, gamma)

        # UPDATE MODEL WITH STATE, NEXT_STATE, ACTION AND REWARD GAINED FROM ACTION
        model = update_model(model, action, reward, state, next_state, episode_time)

        # IF REWARD OF THE ACTION IS -10 (THE LOWEST REWARD) THEN ADD POINT TO PENALTY SCORE
        if reward == -10:
            penalties += 1

        # MAKE PREPARATIONS FOR NEXT LOOP
        state = next_state
        epochs += 1
        episode_time += 1

        # FOR AN ARBITRARY NUMBER OF STEPS, RANDOMLY UPDATE THE Q VALUES MATRIX USING THE MODEL AND TIME
        for _ in range(steps):
            q_values = q_planning(model, q_values, episode_time, time_weight)

    if i % 100 == 0:
        print("Episode: ", i)

end = time.perf_counter()
print("Training completed. Time taken: ", end - start)

# CHECK IF EVAL METHOD WORKS, MORE TIMESTEPS PER EP THAN EXPECTED
print("Results after ", episodes, " episodes:\nAverage timesteps per episode: ", epochs,
      "\nAverage penalties per episode: ", penalties)

# ~~~ EVALUATION OF DYNA-Q AGENT ~~~
total_epochs, total_penalties = 0, 0
episodes = 1

for i in range(episodes):
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
