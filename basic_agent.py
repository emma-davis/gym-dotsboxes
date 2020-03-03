#!/usr/bin/env python
import random
from gym_dotsboxes.dotsboxes_env import DotsBoxesEnv, agent_by_mark, check_game_status, after_action_state, tomark, next_mark


class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            a_total, b_total, a_win, b_win, draw = check_game_status(nstate[0])

            if a_win == True | b_win == True:
                return action

        return random.choice(ava_actions)


def play(max_episode=1):
    episode = 0
    start_mark = 'A'
    env = DotsBoxesEnv()
    agents = [BaseAgent('A'),
              BaseAgent('B')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            env.print_turn(mark)

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            if len(ava_actions) == 0:
                   print("~~~~~ Finished: Draw ~~~~~")
                   break
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.print_result(mark, reward)
                break
            else:
                _, mark = state

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1


if __name__ == '__main__':
    play()
