"""
TRAIN DQN AGENT TO PLAY DOTS AND BOXES AGAINST GREEDY AGENT (FOR NOW)
"""
from gym_dotsboxes.environment import DotsBoxesEnv
from players import ApproxFeaturePlayer, DQNPlayer, GreedyPlayer
import random


def train_model():
    # INITIALISE VARS SUCH AS NUMBER OF GAMES, SAVE MODEL PATH AND LOG PATH
    num_games = 2000  # 20000
    log_file = "C:/Users/Emma/PycharmProjects/deep_q_learning/gym-dotsboxes/training_log.txt"

    # INITIALISE ENVIRONMENT
    start_mark = random.choice(['A', 'B'])
    env = DotsBoxesEnv(start_mark)
    test_env = DotsBoxesEnv(start_mark)

    # TEST: GET A LOOK AT STATE FORMAT
    print(env.get_obs())

    # INITIALISE PLAYERS. ONE WILL BE DQN AND ONE WILL BE SIMPLE GREEDY
    player_A = DQNPlayer(env=env)
    player_A.init_DQN()
    player_B = ApproxFeaturePlayer(env=env)

    # START PLAYING
    for i in range(num_games):

        # RESET ENV AND RANDOMLY SELECT PLAYER TO GO FIRST
        start_mark = random.choice(['A', 'B'])
        env.set_start_mark(start_mark)
        state = env.reset()
        done = False

        while not done:

            # GET CURRENT MARK FROM ENV
            mark = env.get_mark()
            print("CURRENT MARK: ", mark)

            # MOSTLY TO CHECK, SHOULD NOT GET THIS FAR
            if len(env.available_actions) == 0:
                print("~~~~~ Finished: Draw ~~~~~")
                break

            if mark == 'B':
                # GREEDY PLAYER PICKS ACTION THAT COMPLETES
                # SQUARE (IF IT CAN) WHILE AVOIDING SACRIFICING
                # SQUARE
                action = player_B.act(list(state[0]))
            else:
                # DQN AGENT PICKS ACTION BASED ON MAX PREDICTED
                # FROM CURRENT STATE
                action = player_A.act(list(state[0]))

            print("ACTION: " + str(action))

            next_state, reward, done, info = env.step(action)

            # MOVE TO NEXT STATE AND PRINT BOARD
            _, next_mark, next_state_num = next_state
            env.render()

            if done:
                env.print_result()
                break
            else:
                _, mark, state_num = state

            # MOVE TO NEXT STATE AFTER ACTION TAKEN
            state = next_state
            print("NEXT STATE: ", state)
            print("\n\n")

        # BELOW IS WORK IN PROGRESS TO TRACK ACCURACY OF MODEL
        """
        # OUTPUT TEST OF CURRENT TRAINING DQN AGENT TO TRACK TRAINING PROGRESS
        if i % 1000 == 0:
            print("Game {} Test Results".format(i))
            with open(log_file, 'a') as file:
                win_percentage, draw_percentage, loss_percentage = test(test_env, player_A, player_B, num_games)
                file.write(
                    '{},{},{},{}\n'.format(i, win_percentage, draw_percentage, loss_percentage))
                print()

        # TODO: EVERY SO OFTEN, WANT TO STORE THE TRAINED AGENT AND WEIGHTS FOR POSTERITY - STORE MODEL
        """


train_model()
