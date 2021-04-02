import logging
from copy import deepcopy
import gym
from gym import spaces
import numpy as np
CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0

LEFT_PAD = '  '
LOG_FMT = logging.Formatter('%(levelname)s '
                            '[%(filename)s:%(lineno)d] %(message)s',
                            '%Y-%m-%d %H:%M:%S')


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def next_mark(mark):
    return 'X' if mark == 'O' else 'O'


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
    """Return game status by current board status.

    Args:
        board (list): Current board state

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.02, show_number=False):
        self.action_space = spaces.Discrete(NUM_LOC)
        self.action_size=9
        self.observation_space = spaces.Discrete(NUM_LOC)
        self.alpha = alpha
        self.set_start_mark('O')
        self.show_number = show_number
        self.seed()
        self.reset()

    def set_start_mark(self, mark):
        self.start_mark = mark

    def clone(self):
        clone=TicTacToeEnv()
        clone.done=self.done
        clone.current_player=self.current_player
        clone.board=deepcopy(self.board)
        return clone

    def reset(self):
        self.board = [0] * NUM_LOC
        self.current_player = self.start_mark
        self.done = False
        return self._get_obs()

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
          

        loc = action
        if self.done:
            print("CHIPS")
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        self.board[loc] = tocode(self.current_player)
        status = check_game_status(self.board)
        logging.debug("check_game_status board {} mark '{}'"
                      " status {}".format(self.board, self.current_player, status))
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.current_player == 'O' else X_REWARD

        # switch turn
        self.current_player = next_mark(self.current_player)
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        return tuple(self.board), self.current_player

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
        else:
            self._show_board(logging.info)
            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        for j in range(0, 9, 3):
            def mark(i):
                return tomark(self.board[i]) if not self.show_number or\
                    self.board[i] != 0 else str(i+1)
            showfn(LEFT_PAD + '|'.join([mark(i) for i in range(j, j+3)]))
            if j < 6:
                showfn(LEFT_PAD + '-----')

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn('')

    def available_actions(self):
        ret=np.zeros(9)
        if self.done:
            return []
        return [i for i, c in enumerate(self.board) if c == 0]


    def available_actions_boardarg(self,board):
        ret=np.zeros(9)
        return [i for i, c in enumerate(board[0]) if c == 0]




    def getInitBoard(self):
        # return initial board (numpy board)
        self.board = [0] * NUM_LOC
        self.current_player = self.start_mark
        self.done = False
        return self._get_obs()
    def getBoardSize(self):
        return (self.board_size,self.board_size)
    
    def getActionSize(self):
        return 9

    def getNextState(self, board, player, action):
        # If already terminal, then don't do anything
        loc = action
        if self.done:
            return self._get_obs(), -1

        reward = NO_REWARD
        # place
        mark=square_context[player]
        board[loc] = tocode(mark)
        return board, -player

    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }
    def getValidMoves(self, board, player):
        indices=self.get_possible_actions(board, player)
        ret=np.zeros(9)
        status = check_game_status(board)
        if status >= 0:
          return ret
        for i in range(0,9):
            if board[x]!=1 and board[x]!=2:
              ret[x]=1
        return ret
    
    def getGameEnded(self, board, player):
        if self.getValidMoves(board,player)!=[]:
            return 0
        if self.ValidMoves(board,-player)!=[]:
            return 0
        return player
    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return -1*board
    def stringRepresentation(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s
    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def play_action(self, action):
        return self.step(action)

    def get_valid_moves(self, player):
        avail_actions = self.available_actions()
        
        return [(1 if i in avail_actions else 0, (int)(i/3), i%3) for i in range(0,9)    ]

    def get_valid_moves_boardarg(self, player,board):
        avail_actions = self.available_actions_boardarg(board)
        return avail_actions
        return [(1 if i in avail_actions else 0, (int)(i/3), i%3) for i in range(0,9)    ]

    def check_game_over(self, current_player):
        reward=0
        if current_player=='O':
            current_player=1
        else:
            current_player=-1
        status = check_game_status(self.board)
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.current_player == 'O' else X_REWARD
        
        return self.done, current_player*reward 



def set_log_level_by(verbosity):
    """Set log level by verbosity level.

    verbosity vs log level:

        0 -> logging.ERROR
        1 -> logging.WARNING
        2 -> logging.INFO
        3 -> logging.DEBUG

    Args:
        verbosity (int): Verbosity level given by CLI option.

    Returns:
        (int): Matching log level.
    """
    if verbosity == 0:
        level = 40
    elif verbosity == 1:
        level = 30
    elif verbosity == 2:
        level = 20
    elif verbosity >= 3:
        level = 10

    logger = logging.getLogger()
    logger.setLevel(level)
    if len(logger.handlers):
        handler = logger.handlers[0]
    else:
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(LOG_FMT)
    return level


