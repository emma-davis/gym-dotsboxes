#!/usr/bin/env python
import random
from gym_dotsboxes.dotsboxes_env import DotsBoxesEnv, agent_by_mark, check_game_status, after_action_state, to_mark, next_mark
import numpy as np
import random
import time

# TODO: ADD REWARD SYSTEM THAT ENCOURAGES AGENTS TO MAKE BOXES OR INTERCEPT OPPOSING
# BOXES (KINDA DONE?)
# TODO: ADD SEPARATE QVALUES MATRICES FOR PLAYERS A AND B (CHECK BUT THINK IS DONE?)
# TODO: EXPAND BOARD FROM 4 X 4 TO any Y x Y


# ~~~ METHOD 2: Q LEARNING ~~~
# Q-values mapped to (state, action) help agent learn from rewards over time. Q-values updated by equation:
#   Q(state, action) <- (1 - a)Q(state, action) + a(reward + ymaxQ(next state, all actions))
# a is the learning rate - the extent to which q-values updated every iteration)
# y is the discount factor - determines importance we want to give future rewards
 

# BASIC AGENT CLASS
class QLearningAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            a_win, b_win, draw = check_game_status(nstate[0], self.a_score, self.b_score)

            if a_win == True | b_win == True:
                return action

        return random.choice(ava_actions)


# START GAME AND TWO AGENTS WITHIN ENVIRONMENT
def play(max_episode=10):

    env = DotsBoxesEnv()
    
    episode = 0
    start_mark = 'A'
    agents = [QLearningAgent('A'),
              QLearningAgent('B')]

    # CREATES EMPTY Q-VALUES MATRIX THE SIZE OF OBSERVATION X ACTION SPACE
    q_values_a = np.zeros([env.observation_space.n, env.action_space.n])
    q_values_b = np.zeros([env.observation_space.n, env.action_space.n])

    # START TRAINING TIMER
    start = time.perf_counter()

    # SET PARAMETERS
    alpha = 0.1  # THE LEARNING RATE
    gamma = 0.6  # DISCOUNT FACTOR
    epsilon = 0.1  # EXPLORATION VS EXPLOITATION FACTOR

    all_actions = env.available_actions()
    all_epochs = []
    all_penalties = []

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        print(state)

        _, mark, state_num = state
        epochs, penalties, reward = 0, 0, 0
        done = False
        
        while not done:
            env.print_turn(mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            unava_actions = list(set(all_actions) - set(ava_actions))
            q_values = q_values_a

            if mark == "B":
                q_values = q_values_b

            if len(ava_actions) == 0:
                   print("~~~~~ Finished: Draw ~~~~~")
                   break
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # THIS WILL GENERATE ACTIONS BASED ON EXPLORATION VS EXPLOITATION FACTOR - HIGH EPSILON MEANS
            # MORE RANDOM ACTIONS. ELSE, PERFORM ACTION WHERE Q-VALUES HAVE BEEN LEARNED
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()

                if action in unava_actions:
                    print("Random action in unavailable actions!")
                
            else:
                # APPLY MASK TO Q_VALUES TO EXCLUDE UNAVAILABLE ACTIONS
                mask = np.zeros(len(q_values[state_num]), dtype = bool)
                mask[unava_actions] = True
                temp_values = np.ma.array(q_values[state_num], mask = mask)
                print(temp_values)

                
                # GETS POSITION OF MAX REWARD ALONG X AXIS OF STATE IN Q-VALUE MATRIX
                action = np.argmax(temp_values)
                

            next_state, reward, done, info = env.step(action)
            
            # NEWLY ADDED BELOW
            _, next_mark, next_state_num = next_state
            
            env.render()

            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark, state_num = state

            # STORE OLD Q-VALUE AS VAR TO USE IN UPDATE STEP, GET VALUE OF MAX REWARD NEXT STATE
            # FROM Q-VALUE MATRIX TO ALSO USE IN UPDATE STEP
            old_value = q_values[state_num, action]
            next_max = np.max(q_values[next_state_num])

            # CALCULATE NEW Q-VALUE FOR Q(STATE, ACTION) JUST PERFORMED AND REPLACE OLD VALUE
            new_value = ((1 - alpha) * old_value) + (alpha * (reward + gamma * next_max))
            q_values[state_num, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print(mark)
            if mark == "A":
                q_values_a = q_values
            else:
                q_values_b = q_values

                
            #action = agent.act(state, ava_actions)
            #state, reward, done, info = env.step(action)
            

        # SWITCH STARTING AGENT FOR EACH GAME
        #start_mark = next_mark(start_mark)
        episode += 1


    print(q_values_a)
    print("\n\n\n")
    print(q_values_b)
        


if __name__ == '__main__':
    play()
