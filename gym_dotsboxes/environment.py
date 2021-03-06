"""
DOTS AND BOXES ENV USING GYM
"""

# IMPORTS
from functools import reduce
import gym
from gym import spaces
import math
import numpy as np


# ENVIRONMENT CLASS
class DotsBoxesEnv(gym.Env):

    def __init__(self, start_mark):
        self.start_mark = start_mark
        self.state_num = None
        self.b_score = 0
        self.a_score = 0
        self.mark = self.start_mark
        self.done = False
        self.num_actions = 24
        self.grid_size = 4
        self.available_actions = [i for i in range(self.num_actions)]
        self.board = [0] * self.num_actions
        self.state = [0] * self.num_actions
        self.margin = '  '
        self.code_mark_map = {0: '1', 1: 'A', 2: 'B'}
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_actions)
        self.set_start_mark(start_mark)
        self.reset()

    def reset(self):
        # RESET GAME BACK TO EMPTY BOARD AND 0 SCORES
        self.mark = self.start_mark
        self.board = [0] * self.num_actions
        self.b_score = 0
        self.a_score = 0
        self.done = False
        self.available_actions = [i for i in range(self.num_actions)]
        self.action_space = spaces.Discrete(self.num_actions)

        return self.get_obs()

    def agent_by_mark(self, agents, mark):
        # RETURNS AGENT ASSOCIATED WITH MARK
        for agent in agents:
            if agent.mark == mark:
                return agent

    def set_start_mark(self, mark):
        # THIS IS PROBABLY REDUNDANT NOW?
        self.start_mark = mark

    def next_mark(self, mark):
        # RETURN THE MARK OF NEXT PLAYER
        return 'B' if mark == 'A' else 'A'

    def get_mark(self):
        # RETURN CURRENT MARK
        return self.mark

    def to_mark(self, num):
        # RETURN THE MARK REATING TO THE NUMBER
        return self.code_mark_map[int(num)]

    def to_num(self, mark):
        # RETURN THE NUMBER RELATING TO THE MARK
        return 1 if mark == 'A' else 2

    def disable_action(self, action):
        # CALLED TO REMOVE ACTIONS FROM AVAILABLE ACTIONS, THUS REMOVING THEM FROM ACTION SPACE
        self.available_actions.remove(action)

    """
    # THINK THIS IS REDUNDANT?
    def available_actions(self):
        # GET ALL CURRENTLY AVAILABLE ACTIONS
        return [i for i, c in enumerate(self.board) if c == 0]
    """

    def contains(self, action):
        # CHECK IF ACTION IN AVAILABLE ACTIONS
        return action in self.available_actions

    def sample(self):
        # GET RANDOM ACTION FROM AVAILABLE ACTIONS
        return np.random.choice(self.available_actions)

    def get_obs(self):
        # GET VIEW OF BOARD
        return tuple(self.board), self.mark, self.state_num

    def step(self, action):
        # PROGRESS THE GAME VIA CHOSEN ACTION, RELAY NEW BOARD INFO TO CONSOLE

        # REMOVE ACTION FROM ACTION SPACE
        self.disable_action(action)

        square_starts_num = int((self.grid_size - 1) ** 2)

        # GET TOTALS OF PLAYERS BEFORE ACTION COMPLETED IN THIS STEP
        a_old_total = self.a_score
        b_old_total = self.b_score

        square_combos = [[0, 3, 4, 7], [1, 4, 5, 8], [2, 5, 6, 9], [7, 10, 11, 14], [8, 11, 12, 15], [9, 12, 13, 16],
                         [14, 17, 18, 21], [15, 18, 19, 22], [16, 19, 20, 23]]

        # ITERATE THROUGH ALL SQUARE COMBINATIONS TO SEE IF THIS CURRENT ACTION
        # IS THE ONE THAT WILL COMPLETE SQUARE(S), THUS GAINING AGENT REWARDS
        num_squares_won = 0
        for square in square_combos:
            temp = square.copy()
            if action in square:
                temp.remove(action)
                occupied_square_count = 0
                for x in temp:
                    if self.board[x] != 0:
                        occupied_square_count += 1

                # WANT THE SCORE TO BE 3 NOT 4, AS AT THIS POINT THE ACTION
                # HASN'T HAPPENED, SO THE SQUARE SHOULD ONLY HAVE 3 SIDES
                # AND NOT 4
                if occupied_square_count == 3:
                    num_squares_won += 1
                if occupied_square_count > 3:
                    print("~~~~~This shouldn't happen!~~~~~")

        loc = action
        #if self.done:
        #    return self.get_obs(), 0, True, None

        # ADD TO SCORES
        if self.mark == "A":
            self.a_score += num_squares_won
        else:
            self.b_score += num_squares_won

        a_total = self.a_score
        b_total = self.b_score

        self.board[loc] = self.to_num(self.mark)
        a_win, b_win, draw = self.check_game_status(self.board, self.a_score, self.b_score)

        # SET REWARDS (IF ANY). IF A WIN (ASSUMING A IS DQN) THEN 1, IF B WIN THEN -1, IF DRAW THEN 0
        reward = None

        # IF AGENT WON SQUARE THEN ADD SMALL REWARD
        if num_squares_won > 0:
            reward = 0.8

        if a_win:
            self.done = True
            reward = 1
        if b_win:
            self.done = True
            reward = -1
        if draw:
            self.done = True
            reward = 0

        print(" A New Total: ", self.a_score)
        print(" B New Total: ", self.b_score)
        print("A Win: ", a_win, " B Win: ", b_win, " Draw: ", draw, "\n")

        # SEE IF MARK NEEDS TO CHANGE (I.E. IF PLAYER HASN'T WON A SQUARE, THEN THE NEXT
        # TURN IS OPPONENTS TURN
        if (self.to_num(self.mark) == 1) & (a_total - a_old_total > 0):
            print("A wins, no switch in turns.")
        elif (self.to_num(self.mark) == 2) & (b_total - b_old_total > 0):
            print("B wins, no switch in turns.")
        else:
            self.mark = self.next_mark(self.mark)
            print("Switch turns.")

        print("DONE?: ", self.done)

        return self.get_obs(), reward, self.done, None

    def check_game_status(self, board, a_score, b_score):
        # SEE STATE OF GAME IN TERMS OF SCORES, WHO IS WINNING AND IS THERE IS A DRAW

        a_total = a_score
        b_total = b_score
        a_win = False
        b_win = False
        draw = False

        # IF A PLAYER HAS A SCORE OF 5 OR MORE, THEY AUTOMATICALLY WIN AS IT ISN'T POSSIBLE
        # FOR THE OTHER PLAYER TO GET 5 OR HIGHER.
        if a_total >= 5:
            a_win = True

        if b_total >= 5:
            b_win = True

        # CHECK IF THERE ARE ANY MOVES LEFT ON BOARD TO PLAY. IF PRODUCT OF BOARD IS NO LONGER
        # 0 THEN THERE ARE NO MORE MOVES AVAILABLE
        if reduce((lambda x, y: x * y), board) != 0:
            if a_total > b_total:
                a_win = True
            elif b_total > a_total:
                b_win = True
            else:
                draw = True

        return [a_win, b_win, draw]

    def render(self, close=False):
        # PRINTS BOARD UNLESS GUI SHOULD BE CLOSED
        if close:
            return
        self.print_board()

    def print_board(self):
        # DRAW DOTS AND BOXES BOARD. THIS IS MOSTLY FOR SANITY CHECK TO SEE WHAT MODEL DOING.
        # IN PROPER TRAINING PHASE THIS SHOULD BE COMMENTED OUT TO REDUCE COMPUTATIONAL EFFORT
        square_row_steps = int(self.num_actions / (self.grid_size - 1) - 1)
        cutoff_num = self.num_actions - self.grid_size + 1

        for j in range(0, self.num_actions, square_row_steps):
            def mark(i):
                # return self.to_mark(self.board[i]) if not self.show_number or \
                #                                 self.board[i] != 0 else str(i + 1)
                # return self.to_mark(self.board[i] if self.board[i] != 0 else str(i + 1))
                return self.to_mark(self.board[i])

            if j == cutoff_num:
                print(self.margin + 'o' + 'o'.join([mark(i) for i in range(j, j + self.grid_size - 1)]) + 'o')
            else:
                print(self.margin + 'o' + 'o'.join([mark(i) for i in range(j, j + self.grid_size - 1)]) + 'o')
                print(self.margin + ' '.join(
                    [mark(i) for i in range(j + self.grid_size - 1, j + (2 * self.grid_size) - 1)]))

    def print_turn(self, mark):
        # PRINT PLAYER TURN
        print("\n{}'s turn.".format(mark))

    def print_result(self):
        a_win, b_win, draw = check_game_status(self.board, self.a_score, self.b_score)
        if a_win == True:
            print("~~~~~ Finished: Winner is Player A! ~~~~~")
        elif b_win == True:
            print("~~~~~ Finished: Winner is Player B! ~~~~~")

        if draw == True:
            print("~~~~~ Finished: Draw ~~~~~")

        print('')


def check_game_status(board, a_score, b_score):
    # CHECKS STATUS OF BOARD, WHICH INCLUDES STATUS OF WINS/DRAWS AND PLAYER SCORES

    a_total = a_score
    b_total = b_score
    a_win = False
    b_win = False
    draw = False

    # IF A PLAYER HAS A SCORE OF 5 OR MORE, THEY AUTOMATICALLY WIN AS IT ISN'T POSSIBLE
    # FOR THE OTHER PLAYER TO GET 5 OR HIGHER.
    if a_total >= 5:
        a_win = True

    if b_total >= 5:
        b_win = True

    # CHECK IF THERE ARE ANY MOVES LEFT ON BOARD TO PLAY. IF PRODUCT OF BOARD IS NO LONGER
    # 0 THEN THERE ARE NO MORE MOVES AVAILABLE
    if reduce((lambda x, y: x * y), board) != 0:
        if a_total > b_total:
            a_win = True
        elif b_total > a_total:
            b_win = True
        else:
            draw = True

    return [a_win, b_win, draw]
