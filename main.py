from gym_dotsboxes.environment import DotsBoxesEnv
from players import DQNPlayer, UserPlayer, GreedyPlayer
import random

def main():
    # USER VS TRAINED DQN
    start_mark = random.choice(['A', 'B'])
    env = DotsBoxesEnv(start_mark)
    episode = 0
    max_episodes = 30000

    player_A = DQNPlayer(env=env)
    player_B = GreedyPlayer(env=env)

    players = [player_A, player_B]

    while episode < max_episodes:

        start_mark = random.choice(['A', 'B'])
        env.set_start_mark(start_mark)
        state = env.reset()

        _, mark, state_num = state
        epochs = 0
        done = False

        while not done:

            # LET CONSOLE KNOW WHO'S TURN IT IS
            mark = env.get_mark()
            env.print_turn(mark)

            player = agent_by_mark(players, mark)
            ava_actions = env.available_actions()
            unava_actions = list(set(env.available_actions) - set(ava_actions))

            if len(ava_actions) == 0:
                print("~~~~~ Finished: Draw ~~~~~")
                break

            # PLAYERS CHOOSE THEIR ACTIONS ACCORDING TO THEIR OWN POLICIES
            if player == player_B:
                # IF PLAYER B THEN GREEDY
                action = player_B.act()
            else:
                # IF PLAYER A THEN DQN
                action = player_A.act()

            # IMPLEMENT ACTION ON BOARD AND DISPLAY
            next_state, reward, done, info = env.step(action)
            _, next_mark, next_state_num = next_state
            env.render()

            # CHECK WIN STATE OF GAME
            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark, state_num = state

        # SWITCH STARTING AGENT FOR EACH GAME
        start_mark = next_mark(start_mark)
        episode += 1
