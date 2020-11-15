"""
TRAIN DQN AGENT TO PLAY DOTS AND BOXES AGAINST GREEDY AGENT (FOR NOW)
"""
from gym_dotsboxes.environment import DotsBoxesEnv
from players import GreedyPlayer, DQNPlayer
import random


def test(test_env, training_player, test_player, num_games):
    """
    Tests an environment against a test agent
    :param test_env: game environment
    :param train: learning agent
    :param test: testing agent
    :param test_games: number of test games to play (integer)
    :return: evaluation metrics
    """

    # If the agent is set to learn, make sure it's switched back on before the function completes
    restart_learn = training_player.learning == True

    training_player.learning = False
    test_env.player1 = training_player
    test_env.player2 = test_player
    test_winners = []
    games = []
    states = []
    scores = []

    for test_game in range(num_games):
        # print("Test Game: {}".format(test_game))
        game, winner, game_length, state_log, final_score = test_env.play(log=True)
        test_winners.append(winner)
        games.append(game)
        states.append(state_log)
        scores.append(final_score)

        # SWITCH PLAYERS
        p1 = test_env.player1
        p2 = test_env.player2
        test_env.player1 = p2
        test_env.player2 = p1

    win_percentage = float(test_winners.count(training_player.name)) / num_games
    draw_percentage = float(test_winners.count('None')) / num_games
    loss_percentage = float(test_winners.count(test.name)) / num_games

    print("Current win percentage over agent {}: {:.2f}%".format(test.name, win_percentage * 100))
    print("Current draw percentage over agent {}: {:.2f}%".format(test.name, draw_percentage * 100))
    print("Current loss percentage over agent {}: {:.2f}%".format(test.name, loss_percentage * 100))

    if restart_learn:
        training_player.learning = True

    return win_percentage, draw_percentage, loss_percentage


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
    player_B = GreedyPlayer(env=env)

    # TODO: GET PATH WHERE MODEL IS SAVED AND LOADED FROM

    # START PLAYING
    for i in range(num_games):

        # RESET ENV AND RANDOMLY SELECT PLAYER TO GO FIRST
        start_mark = random.choice(['A', 'B'])
        env.set_start_mark(start_mark)
        state = env.reset()
        print(state)
        done = False

        # TODO: WRITE PLAY FUNCTION GET ALL AVAILABLE ACTIONS AS ALL ACTIONS, AND SET AVAILABLE ACTIONS AS ALL ACTIONS
        #  AT START OF GAME

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
                action = player_B.act()
            else:
                # DQN AGENT PICKS ACTION BASED ON MAX PREDICTED
                # FROM CURRENT STATE
                action = player_A.act()

            print("ACTION: " + str(action))

            # TODO: THIS INFO NEEDS TO BE FED INTO THE UPDATING WEIGHTS STEP TO TRAIN MODEL - FIGURE OUT HOW TO DO THIS

            next_state, reward, done, info = env.step(action)

            # MOVE TO NEXT STATE AND PRINT BOARD
            _, next_mark, next_state_num = next_state
            env.render()

            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark, state_num = state

            # MOVE TO NEXT STATE AFTER ACTION TAKEN
            state = next_state
            print("\n\n")

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
