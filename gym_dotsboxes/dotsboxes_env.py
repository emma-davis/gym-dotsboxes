from functools import reduce
import gym
from gym import spaces
import logging

CODE_MARK_MAP = {0: '-', 1: 'A', 2: 'B'}
NUM_ACTIONS = 24
GRID_SIZE = 4 # NUM DOTS ALONG X AND Y (SQUARE GRID FOR NOW)
POS_REWARD = 1
NEG_REWARD = -1
NO_REWARD = 0

MARGIN = '  '


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'A' else 2


def next_mark(mark):
    return 'B' if mark == 'A' else 'A'


def agent_by_mark(agents, mark):
    for agent in agents:
        if agent.mark == mark:
            return agent


def after_action_state(state, action):
    """Execute an action and returns resulted state.
    Args:
        state (tuple): Board status + mark
        action (int): Action to run
    Returns:
        tuple: New state
    """

    board, mark = state
    nboard = list(board[:])
    nboard[action] = tocode(mark)
    nboard = tuple(nboard)
    return nboard, next_mark(mark)


def check_game_status(board):
    """
    Returns a list of [a_total, b_total, a_win, b_win], where each total is the number
    of complete squares each player has and each win is a boolean (T/F) stating if that
    player has won yet.
    """

    start_x = [0, 1, 2, 7, 8, 9, 14, 15, 16]
    a_total = 0
    b_total = 0
    a_win = False
    b_win = False
    draw = False

    # ITERATE THROUGH 1 (PLAYER A) AND 2 (PLAYER B) ON BOARD TO CHECK
    # FOR COMPLETE SQUARES. SQUARE RULE IS [X, X+3, X+4, X+7]
    for t in [1, 2]:
        for x in start_x:
            if [board[x], board[x + 3], board[x + 4], board[x + 7]] == [t]*4:
                if t == 1:
                    a_total += 1
                else:
                    b_total += 1
        
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
        
 
    return [a_total, b_total, a_win, b_win, draw]


class DotsBoxesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, show_number=False):
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Discrete(NUM_ACTIONS)
        self.set_start_mark('A')
        self.show_number = show_number
        self.seed()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def reset(self):
        self.board = [0] * NUM_ACTIONS
        self.mark = self.start_mark
        self.done = False
        return self.get_obs()

    def step(self, action):
        """Step environment by action.
        Args:
            action (int): Location
        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self.get_obs(), 0, True, None

        reward = NO_REWARD
        
        # CHECK BOARD BEFORE ACTION, THEN PLACE ACTION ON BOARD AND CHECK
        # BOARD AGAIN TO COMPARE
        a_old_total, b_old_total, a_old_win, b_old_win, old_draw = check_game_status(self.board)
        
        self.board[loc] = tocode(self.mark)
        a_total, b_total, a_win, b_win, draw = check_game_status(self.board)

        if a_win == True | b_win == True | draw == True:
            self.done = True

        print("\nA Old Total: ", a_old_total, " A New Total: ", a_total)
        print("B Old Total: ", b_old_total, " B New Total: ", b_total)
        print("A Win: ", a_win, " B Win: ", b_win, " Draw: ", draw, "\n")

        
        # CHECK IF A OR B HAVE WON A SQUARE IN THIS STEP, IF THEY HAVE THEN SET REWARDS
        # ACCORDINGLY AND KEEP CURRENT MARK SAME AS WINNER TO GIVE THEM ANOTHER TURN

        # IF CURRENT MARK IS A AND A HAS GAINED ONE SQUARE, THEN REWARD A
        # DO SAME FOR B, ELSE IF CURRENT MARKS HAVE NOT WON SQUARES THEN MOVE ON
        # TO NEXT MARK
        if (tocode(self.mark) == 1) & (a_total - a_old_total > 0):
            reward = POS_REWARD
        elif (tocode(self.mark) == 2) & (b_total - b_old_total > 0):
            reward = POS_REWARD
        else:
            self.mark = next_mark(self.mark)

        """
        # CURRENT MARK IS A AND NOT GAINED 
            elif a_total == a_old_total:
                self.mark = next_mark(self.mark)


        elif tocode(self.mark) == 2:
            if b_total - b_old_total > 0:
                reward = POS_REWARD

            elif b_total == a_old_total:
                self.mark = next_mark(self.mark)
        """

        return self.get_obs(), reward, self.done, None

    def get_obs(self):
        return tuple(self.board), self.mark


    def render(self, close=False):
        if close:
            return
        self.print_board()


    def print_board(self):
        """Draw dots and boxes board."""
        # TODO: MAKE THIS MORE DYNAMIC AT SOME POINT TO CATER FOR DIFFERENT
        # BOARD SIZES
        for j in range(0, NUM_ACTIONS, 7):
            def mark(i):
                return tomark(self.board[i]) if not self.show_number or\
                    self.board[i] != 0 else str(i+1)
            if j == 21:
                print(MARGIN + 'o' + 'o'.join([mark(i) for i in range(j, j+3)]) + 'o')
            else:
                print(MARGIN + 'o' + 'o'.join([mark(i) for i in range(j, j+3)]) + 'o')
                print(MARGIN + ' '.join([mark(i) for i in range(j+3, j+7)]))


    def print_turn(self, mark):
        print("\n{}'s turn.".format(mark))


    def print_result(self, mark, reward):
        a_total, b_total, a_win, b_win, draw = check_game_status(self.board)
        if a_win == True:
            print("~~~~~ Finished: Winner is Player A! ~~~~~")
        elif b_win == True:
            print("~~~~~ Finished: Winner is Player B! ~~~~~")

        if draw == True:
            print("~~~~~ Finished: Draw ~~~~~")
        
        print('')


    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]
