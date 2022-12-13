import numpy as np
import copy
import random
from typing import Optional
from typing_extensions import Literal
import warnings
import torch
from Node import Node
from ChessBoard import ChessBoard
from Model import Net
'''
source: https://github.com/TsrmKumoko/gomoku_mcts
'''

class Agent(object):
    '''
    # Agent
    An agent class to play gomoku with human using MCTS.
    '''
    def __init__(
        self,
        size: int = 15,
        win_len: int = 5,
        max_searches: int = 10000,
        net: Optional[Net] = None
    ) -> None:
        '''
        Initialize an agent.
        ## Parameters\n
        size: int, optional
            Size of the gomoku board. Default is 15, which is the standard size.
            Don't passed a number greater than 15.
        win_len: int, optional
            Number of stones in a line needed to win a game. Default is 5.
        max_searches: int, optional
            Number of games the agent plays with itself after a human's move.
            Default value is 10000.
        '''
        self.root = Node()
        self.current_node = self.root
        self.current_node.visits += 1
        self.board_size = size
        self.win_len = win_len
        self.chess_board = ChessBoard(size=size, win_len=win_len)
        self.max_searches = max_searches
        self.net = net
        if net is not None:
            net.eval()
        return

    def update_root(self, move:tuple) -> None:
        '''
        When a move is decided, this method should be called to
        change the root node to the current move.
        If the move has already been searched, it will be set as the root,
        otherwise a new node will be created.
        ## Parameters\n
        move: tuple
            Coordinate of the stone last played.
        '''
        # Please change os.system('clear') to os.system('cls') if your system is Windows!
        for child in self.current_node.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                self.visit(self.root)
                # os.system('clear')
                # self.chess_board.display_board()
                # print(move)
                return
        node = Node(
            color = -self.current_node.color,
            depth = self.current_node.depth + 1,
            move = move
        )
        self.root = node
        self.root.parent = None
        self.visit(self.root)
        # os.system('clear')
        # self.chess_board.display_board()
        # print(move)
        return

    def visit(self, node:Node) -> None:
        '''
        Visit the given node. This method will only be called when searching.
        If you want **place a stone** on the board, please use the `update_root` method.
        ## Parameters\n
        node: Node
            The node to be visited.
        '''
        self.current_node = node
        self.current_node.visits += 1
        self.chess_board.play_stone(node.move)

    def chosen_child(self) -> Node:
        '''
        ## Returns\n
        out: Node
            The child of current node with the highest UCB value.
        '''
        zero_visits = []
        total_visits = 0
        for child in self.current_node.children:
            total_visits += child.visits
            if child.visits == 0:
                zero_visits.append(child)
        if zero_visits != []:
            return random.choice(zero_visits)
        else:
            UCB = lambda value, visits, move: value / visits + \
                np.sqrt(2 * np.log(total_visits) / visits) * \
                self.current_node.prob[move]
            UCB_list = [
                UCB(child.value, child.visits, child.move)
                for child in self.current_node.children
            ]
            idxmax = UCB_list.index(max(UCB_list))
        return self.current_node.children[idxmax]

    def best_child(self) -> Node:
        '''
        ## Returns\n
        out: Node
            The child of the root node who has the most visits.
            We simply assume that more visits means better node.
        '''
        for child in self.current_node.children:
            if child.is_ended:
                return child
        visits_list = [
            child.visits for child in self.current_node.children
        ]
        idxmax = visits_list.index(max(visits_list))
        return self.current_node.children[idxmax]

    def expand_current_node(self) -> None:
        '''
        Expand a visited but has no child node with possible moves.
        '''
        if self.current_node.children != []:
            warnings.warn('This node is already expanded.', Warning, 2)
        else:
            vacancies = self.chess_board.adjacent_vacancies()
            if self.net is not None:
                if self.current_node.prob is None:
                    feature = self.net.preprocess(self.chess_board)
                    with torch.no_grad():
                        p, v = self.net(feature)
                    self.current_node.prob = self.net.normalize_prob(p[0], vacancies)
                    self.current_node.eval_value = v.item()
            else:
                self.current_node.prob = torch.ones((self.board_size, self.board_size))
            for move in vacancies:
                child = Node(
                    self.current_node,
                    -self.current_node.color,
                    self.current_node.depth + 1,
                    move
                )
                self.current_node.children.append(child)
        return

    def roll_out(self) -> Literal[1, 0, -1]:
        '''
        Randomly play the rest of the game with itself and return the reward.
        ## Returns\n
        out: Literal[1, 0, -1]
            The reward the leaf node gets. `1` for a win, `-1` for a lose and `0` for a draw.
        '''
        while not self.chess_board.is_ended():
            vacancies = self.chess_board.adjacent_vacancies()
            loc = random.choice(list(vacancies))
            self.chess_board.play_stone(loc)
        if self.chess_board.winner == self.current_node.color:
            return 1
        elif self.chess_board.winner == -self.current_node.color:
            return -1
        else:
            return 0
    
    def eval_value(self) -> float:
        if self.chess_board.is_ended():
            if self.chess_board.winner == self.current_node.color:
                return 1
            elif self.chess_board.winner == -self.current_node.color:
                return -1
        if self.current_node.eval_value is None:
            feature = self.net.preprocess(self.chess_board)
            with torch.no_grad():
                p, v = self.net(feature)
            self.current_node.prob = self.net.normalize_prob(p[0], self.chess_board.adjacent_vacancies())
            self.current_node.eval_value = v.item()
        return self.current_node.eval_value

    def back_propagate(self, reward:Literal[1, 0, -1]) -> None:
        '''
        Update all values on the way back to the root.
        ## Parameters\n
        reward: Literal[1, 0, -1]
            The reward after rollout.
        '''
        while self.current_node.parent != None:
            self.current_node.value += reward
            self.current_node = self.current_node.parent
            reward = -reward
        return

    def search(self, move:tuple) -> None:
        '''
        Search the best move according to current state and play a stone.
        '''
        self.update_root(move)
        if self.chess_board.is_ended():
            return
        chess_board_copy = copy.deepcopy(self.chess_board)
        for _ in range(self.max_searches):
            # print('Searching: ', end='')
            # print(round(float(_) * 100 / self.max_searches), end='%\r')
            is_ended = False
            while self.current_node.children != [] and not is_ended:
                self.visit(self.chosen_child())
                is_ended = self.chess_board.is_ended()
            if self.current_node.visits > 0:
                if is_ended:
                    self.current_node.ended = True
                else:
                    if self.current_node.children == []:
                        self.expand_current_node()
                    self.visit(self.chosen_child())
            else:
                raise NotImplementedError
            if is_ended:
                self.back_propagate(self.chess_board.winner == self.current_node.color)
            elif self.net is not None:
                self.back_propagate(self.eval_value())
            else:
                self.back_propagate(self.roll_out())
            
            self.chess_board = copy.deepcopy(chess_board_copy)
            #self.chess_board.moves = self.chess_board.moves[0:self.root.depth]
            #self.chess_board.board = [
            #    [0 for _ in range(self.board_size)]
            #    for _ in range(self.board_size)
            #]
            #color = 1
            #for move in self.chess_board.moves:
            #    self.chess_board.board[move[0]][move[1]] = color
            #    color = -color
            #self.chess_board.now_playing = -self.root.color
            #self.chess_board.winner = 0
        best_move = self.best_child().move
        if hasattr(self, 'episode'):
            feature = Net.preprocess(self.chess_board).bool()
            prob = torch.zeros((self.board_size, self.board_size))
            total_visits = 0
            for child in self.current_node.children:
                total_visits += child.visits
                prob[child.move] = child.visits
            prob /= total_visits
            self.episode.append((feature, prob))
        self.update_root(best_move)
        return best_move