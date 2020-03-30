from functools import reduce
import gym
from gym import spaces
import logging
import numpy as np

# SET UP VARS
CODE_MARK_MAP = {0: '-', 1: 'A', 2: 'B'}
NUM_ACTIONS = 24
GRID_SIZE = 4 # NUM DOTS ALONG X AND Y (SQUARE GRID FOR NOW)
POS_REWARD = 1
NEG_REWARD = -1
NO_REWARD = 0
MARGIN = '  '


# RETURN THE MARK RELATING TO THE NUMBER
def to_mark(num):
    return CODE_MARK_MAP[num]


# RETURN THE NUMBER RELATING TO THE MARK
def to_num(mark):
    return 1 if mark == 'A' else 2


# RETURN THE MARK OF NEXT PLAYER
def next_mark(mark):
    return 'B' if mark == 'A' else 'A'


# RETURNS AGENT ASSOCIATED WITH MARK
def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


# EXECUTES ACTION ON BOARD AND RETURNS RESULTING STATE
def after_action_state(state, action):
    board, mark = state
    nboard = list(board[:])
    nboard[action] = to_num(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


# CHECKS STATUS OF BOARD, WHICH INCLUDES STATUS OF WINS/DRAWS AND PLAYER SCORES
def check_game_status(board, a_score, b_score):
    """
    Returns a list of [a_total, b_total, a_win, b_win, draw], where each total is the number
    of complete squares each player has and each win is a boolean (T/F) stating if that
    player has won yet, or if players have drawn.
    
    """

    start_x = [0, 1, 2, 7, 8, 9, 14, 15, 16]
    a_total = a_score
    b_total = b_score
    a_win = False
    b_win = False
    draw = False

    # ~~~~~ SCORING IS NO LONGER DONE IN THIS FUNCTION, TODO: REMOVE THE BELOW ~~~~~
    # ~~~~~ ONCE SCORING HAS BEEN PLACED ELSEWHERE ~~~~~

    ## ITERATE THROUGH 1 (PLAYER A) AND 2 (PLAYER B) ON BOARD TO CHECK
    ## FOR COMPLETE SQUARES. SQUARE RULE IS [X, X+3, X+4, X+7]
    #for t in [1, 2]:
    #    for x in start_x:
    #        if [board[x], board[x + 3], board[x + 4], board[x + 7]] == [t]*4:
    #            if t == 1:
    #                a_total += 1
    #            else:
    #                b_total += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
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


# ENVIRONMENT CLASS
class DotsBoxesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_number=False):
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Discrete(NUM_ACTIONS)
        self.set_start_mark('A')
        self.start_state = np.random.randint(0, 24)
        self.show_number = show_number
        self.seed()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def reset(self):
        self.board = [0] * NUM_ACTIONS
        self.mark = self.start_mark
        self.a_score = 0
        self.b_score = 0
        self.state_num = self.start_state
        self.done = False
        return self.get_obs()

    def step(self, action):
        """
        Returns observation of board once action is performed on it, the reward gained by agent,
        whether the game is done as a result of the action and any extra information (currently set
        as None).
        """

        # GET TOTALS OF PLAYERS BEFORE ACTION COMPLETED IN THIS STEP
        a_old_total = self.a_score
        b_old_total = self.b_score

        # ALL SQUARE COMBINATIONS POSSIBLE
        square_combos = [[0,3,4,7], [1,4,5,8], [2,5,6,9],
                         [7,10,11,14], [8,11,12,15], [9,12,13,16],
                         [14,17,18,21], [15,18,19,22], [16,19,20,23]]

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
        
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self.get_obs(), 0, True, None

        reward = num_squares_won * 1 # ARBITRARY SCORE FOR NOW

        # ADD TO SCORES
        if self.mark == "A":
            self.a_score += num_squares_won
        else:
            self.b_score += num_squares_won

        a_total = self.a_score
        b_total = self.b_score

        # ~~~~~ TODO: THIS MAY NOT NEED TO BE DONE?
        # CHECK BOARD BEFORE ACTION, THEN PLACE ACTION ON BOARD AND CHECK
        # BOARD AGAIN TO COMPARE
        #a_old_win, b_old_win, old_draw = check_game_status(self.board, self.a_score, self.b_score)
        
        self.board[loc] = to_num(self.mark)
        a_win, b_win, draw = check_game_status(self.board, self.a_score, self.b_score)

        if a_win == True | b_win == True | draw == True:
            self.done = True

        print(" A New Total: ", self.a_score)
        print(" B New Total: ", self.b_score)
        print("A Win: ", a_win, " B Win: ", b_win, " Draw: ", draw, "\n")

        # ~~~~~ START OF REWARD SYSTEM ~~~~~
        # THIS SECTION WILL CHANGE FREQUENTLY DEPENDING ON WHICH REWARD SYSTEMS WORK
        # BEST. CURRENTLY GIVE SMALL REWARDS FOR SEMI-COMPLETE BOXES, LARGE REWARDS FOR
        # FULLY COMPLETE BOXES AND SOME REWARDS FOR INTERCEPTED BOXES (WHERE THE OPPOSING
        # PLAYER HAS 3 SIDES DOWN AND THE PLAYER INTERCEPTS THE LAST SIDE)

        # NEED TO HAVE SYSTEM WHERE AGENTS WANT TO COMPLETE BOXES, EITHER MADE BY THEMSELVES
        # THEIR OPPONENT OR A COMBINATION OF THE TWO.

        # WANT TO PENALISE AGENTS FOR LETTING OPPONENTS GET BIG CHAINS
        
        # CHECK IF A OR B HAVE WON A SQUARE IN THIS STEP, IF THEY HAVE THEN SET REWARDS
        # ACCORDINGLY AND KEEP CURRENT MARK SAME AS WINNER TO GIVE THEM ANOTHER TURN

        # IF CURRENT MARK IS A AND A HAS GAINED ONE SQUARE, THEN REWARD A
        # DO SAME FOR B, ELSE IF CURRENT MARKS HAVE NOT WON SQUARES THEN MOVE ON
        # TO NEXT MARK
        # TODO: THIS IS PROBABLY REDUNDANT, AS WE WILL HAVE SEPARATE QVALUES FOR
        # EACH AGENT
        if (to_num(self.mark) == 1) & (a_total - a_old_total > 0):
            reward = POS_REWARD
        elif (to_num(self.mark) == 2) & (b_total - b_old_total > 0):
            reward = POS_REWARD
        else:
            self.mark = next_mark(self.mark)

        return self.get_obs(), reward, self.done, None
    
    
    # RETURNS CURRENT STATE OF BOARD, MARK OF CURRENT PLAYER AND STATE OF CURRENT PLAYER
    def get_obs(self):
        return tuple(self.board), self.mark, self.state_num


    # PRINTS BOARD UNLESS GUI SHOULD BE CLOSED
    def render(self, close=False):
        if close:
            return
        self.print_board()


    # DEFINES THE FORMATTING OF BOARD AND PRINTS CURRENT STATE
    def print_board(self):
        """
        Draw dots and boxes board.
        
        """
        # TODO: MAKE THIS MORE DYNAMIC AT SOME POINT TO CATER FOR DIFFERENT
        # BOARD SIZES
        for j in range(0, NUM_ACTIONS, 7):
            def mark(i):
                return to_mark(self.board[i]) if not self.show_number or\
                    self.board[i] != 0 else str(i+1)
            if j == 21:
                print(MARGIN + 'o' + 'o'.join([mark(i) for i in range(j, j+3)]) + 'o')
            else:
                print(MARGIN + 'o' + 'o'.join([mark(i) for i in range(j, j+3)]) + 'o')
                print(MARGIN + ' '.join([mark(i) for i in range(j+3, j+7)]))


    def print_turn(self, mark):
        print("\n{}'s turn.".format(mark))


    def print_result(self, mark, reward):
        a_win, b_win, draw = check_game_status(self.board, self.a_score, self.b_score)
        if a_win == True:
            print("~~~~~ Finished: Winner is Player A! ~~~~~")
        elif b_win == True:
            print("~~~~~ Finished: Winner is Player B! ~~~~~")

        if draw == True:
            print("~~~~~ Finished: Draw ~~~~~")
        
        print('')


    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]
