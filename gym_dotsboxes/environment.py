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
        self.num_actions = 24  # HARDCODED FOR NOW, MAKE FLEXIBLE LATER
        self.grid_size = 4  # ALSO HARDCODED FOR NOW, MAKE FLEXIBLE LATER
        self.available_actions = [i for i in range(self.num_actions)]
        self.board = [0] * self.num_actions  # HARDCODED FOR NOW
        self.state = [0] * self.num_actions#np.zeros([self.grid_size, self.grid_size, 4])
        self.margin = '  '
        self.code_mark_map = {0: '1', 1: 'A', 2: 'B'}
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_actions)
        self.set_start_mark(start_mark)
        self.reset()

    def reset(self):
        # RESET GAME BACK TO EMPTY BOARD AND 0 SCORES
        self.mark = self.start_mark
        self.board = [0] * self.num_actions # HARDCODED FOR NOW
        #self.state = np.zeros([self.grid_size, self.grid_size, 4])
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
        """
        Returns observation of board once action is performed on it, the reward gained by agent,
        whether the game is done as a result of the action and any extra information (currently set
        as None).
        """

        # REMOVE ACTION FROM ACTION SPACE
        self.disable_action(action)

        square_starts_num = int((self.grid_size - 1) ** 2)
        square_starts_per_row = int(math.sqrt(square_starts_num))
        square_row_steps = int(self.num_actions / (self.grid_size - 1) - 1)

        # GET TOTALS OF PLAYERS BEFORE ACTION COMPLETED IN THIS STEP
        a_old_total = self.a_score
        b_old_total = self.b_score

        square_starts = []
        square_combos = []

        # CREATE A LIST OF ALL THE STARTING NUMBER EDGES OF ALL SQUARES
        # POSSIBLE FOR SPECIFIC GRID
        for i in range(0, self.num_actions - square_row_steps, square_row_steps):
            for j in range(0, square_starts_per_row):
                square_starts.append(i + j)

        for i in square_starts:
            square_side_step = self.grid_size - 1
            square_combos.append([i, i + square_side_step, i + square_side_step + 1,
                                  i + (2 * square_side_step) + 1])

        # ITERATE THROUGH ALL SQUARE COMBINATIONS TO SEE IF THIS CURRENT ACTION
        # IS THE ONE THAT WILL COMPLETE SQUARE(S), THUS GAINING AGENT REWARDS
        num_squares_won = 0
        for square in square_combos:
            if action in square:
                square.remove(action)
                occupied_square_count = 0
                for x in square:
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
        if self.done:
            return self.get_obs(), 0, True, None

        # ADD TO SCORES
        if self.mark == "A":
            self.a_score += num_squares_won
        else:
            self.b_score += num_squares_won

        a_total = self.a_score
        b_total = self.b_score

        self.board[loc] = self.to_num(self.mark)
        a_win, b_win, draw = self.check_game_status(self.board, self.a_score, self.b_score)

        if a_win == True | b_win == True | draw == True:
            self.done = True

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

        return self.get_obs(), 0, self.done, None

    def check_game_status(self, board, a_score, b_score):
        """
        Returns a list of [a_win, b_win, draw], where each total is the number
        of complete squares each player has and each win is a boolean (T/F) stating if that
        player has won yet, or if players have drawn.

        """

        a_total = a_score
        b_total = b_score
        a_win = False
        b_win = False
        draw = False

        # TODO: CHANGE BELOW TO DEAL WITH ALL GRID SIZES, NOT JUST 4X4 (AS IS CURRENTLY)

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
        """
        Draw dots and boxes board.

        # TODO: MAKE THIS MORE DYNAMIC AT SOME POINT TO CATER FOR DIFFERENT
        # BOARD SIZES
        """
        square_row_steps = int(self.num_actions / (self.grid_size - 1) - 1)
        cutoff_num = self.num_actions - self.grid_size + 1

        for j in range(0, self.num_actions, square_row_steps):
            def mark(i):
                #return self.to_mark(self.board[i]) if not self.show_number or \
                #                                 self.board[i] != 0 else str(i + 1)
                #return self.to_mark(self.board[i] if self.board[i] != 0 else str(i + 1))
                return self.to_mark(self.board[i])

            if j == cutoff_num:
                print(self.margin + 'o' + 'o'.join([mark(i) for i in range(j, j + self.grid_size - 1)]) + 'o')
            else:
                print(self.margin + 'o' + 'o'.join([mark(i) for i in range(j, j + self.grid_size - 1)]) + 'o')
                print(self.margin + ' '.join([mark(i) for i in range(j + self.grid_size - 1, j + (2 * self.grid_size) - 1)]))

    def print_turn(self, mark):
        # PRINT PLAYER TURN
        print("\n{}'s turn.".format(mark))
