from typing import Set, List
from typing_extensions import Literal
from collections import defaultdict

class ChessBoard(object):
    '''
    Chess Board
    A gomoku board class.
    '''
    def __init__(self, size:int=15, win_len:int=5) -> None:
        '''
        Initialize a board.
        ## Parameters\n
        size: int, optional
            Size of the gomoku board. Default is 15, which is the standard size.
            Don't passed a number greater than 15.
        win_len: int, optional
            Number of stones in a line needed to win a game. Default is 5.
        '''
        self.size = size
        self.win_len = win_len
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.count = [[0 for _ in range(size)] for _ in range(size)]
        self.link3 = {c: set() for c in [1, -1]}
        self.link4 = {c: set() for c in [1, -1]}
        self.link5 = {c: set() for c in [1, -1]}
        self.par_link3 = {c: set() for c in [1, -1]}
        self.par_link4 = {c: set() for c in [1, -1]}
        self.cross = {c: set() for c in [1, -1]}
        self.par_cross = {c: set() for c in [1, -1]}
        self.dir = {c: {} for c in [1, -1]}
        self.par_dir = {c: defaultdict(dict) for c in [1, -1]}
        self.moves: List[tuple] = []
        self.now_playing: Literal[1, -1] = 1
        self.winner = 0
    
    def is_inside(self, move:tuple) -> bool:
        i, j = move
        is_inside = i >= 0 and i < self.size and j >= 0 and j < self.size
        return is_inside

    def is_legal(self, move:tuple) -> bool:
        '''
        Judge whether a stone can be placed at given coordinate.
        ## Parameters\n
        move: tuple
            The coordinate of move about to be judged.
        '''
        is_inside = self.is_inside(move)
        if not is_inside:
            return False
        is_vacancy = self.board[move[0]][move[1]] == 0
        return is_vacancy

    def play_stone(self, move:tuple) -> None:
        '''
        Play a stone at the given coordinate.
        ## Parameters\n
        move: tuple
            The coordinate of move to be played.
        '''
        if not self.is_legal(move):
            raise ValueError(f'Cannot play a stone at {move}.')
        else:
            self.board[move[0]][move[1]] = self.now_playing
            self.moves.append(move)
            self.update_count(move)
            self.now_playing = -self.now_playing
        return
    
    def update_count(self, move:tuple) -> None:
        for player in (1, -1):
            for group in (self.link3, self.link4, self.link5, self.cross, self.par_cross):
                if move in group[player]:
                    group[player].remove(move)
            for group in (self.par_link3, self.par_link4):
                if move in group[player]:
                    group[player].remove(move)
                    self.par_dir[player][move].clear()
        for i in range(2):
            for j in range(-1, 2):
                if i == 0 and j < 1:
                    continue
                l_max, r_max = 0, 0
                l_ext, r_ext = 0, 0
                l_par, r_par = 0, 0
                while True:
                    x = move[0] + (l_max - 1) * i
                    y = move[1] + (l_max - 1) * j
                    if not self.is_inside((x, y)) or self.board[x][y] != self.now_playing:
                        if self.is_inside((x, y)):
                            if self.board[x][y] == 0:
                                x -= i
                                y -= j
                                while self.is_inside((x, y)) and self.board[x][y] == self.now_playing:
                                    l_ext += 1
                                    x -= i
                                    y -= j
                                if l_ext == 0 and self.is_inside((x, y)) and self.board[x][y] == 0:
                                    x -= i
                                    y -= j
                                    while self.is_inside((x, y)) and self.board[x][y] == self.now_playing:
                                        l_par += 1
                                        x -= i
                                        y -= j
                        break
                    l_max -= 1
                while True:
                    x = move[0] + (r_max + 1) * i
                    y = move[1] + (r_max + 1) * j
                    if not self.is_inside((x, y)) or self.board[x][y] != self.now_playing:
                        if self.is_inside((x, y)):
                            if self.board[x][y] == 0:
                                x += i
                                y += j
                                while self.is_inside((x, y)) and self.board[x][y] == self.now_playing:
                                    r_ext += 1
                                    x += i
                                    y += j
                                if r_ext == 0 and self.is_inside((x, y)) and self.board[x][y] == self.now_playing:
                                    x += i
                                    y += j
                                    while self.is_inside((x, y)) and self.board[x][y] == self.now_playing:
                                        r_par += 1
                                        x += i
                                        y += j
                        break
                    r_max += 1

                cnt = r_max - l_max + 1
                for k in range(l_max, r_max + 1):
                    x = move[0] + k * i
                    y = move[1] + k * j
                    self.count[x][y] = max(self.count[x][y], cnt)
                
                l_x = move[0] + (l_max - 1) * i
                l_y = move[1] + (l_max - 1) * j
                ll_x = move[0] + (l_max - 2) * i
                ll_y = move[1] + (l_max - 2) * j
                r_x = move[0] + (r_max + 1) * i
                r_y = move[1] + (r_max + 1) * j
                rr_x = move[0] + (r_max + 2) * i
                rr_y = move[1] + (r_max + 2) * j

                if self.is_inside((l_x, l_y)) and self.board[l_x][l_y] == 0:
                    for group in (self.par_link3, self.par_link4):
                        for player in (1, -1):
                            # remove same direction adjacant stone
                            if (l_x, l_y) in group and (i, j) in self.par_dir[player][(l_x, l_y)]:
                                group[player].remove((l_x, l_y))
                                self.par_dir[player][(l_x, l_y)].pop((i, j))
                                if len(self.par_dir[player]) + ((l_x, l_y) in self.dir[player]) < 2:
                                    if (l_x, l_y) in self.par_cross[self.now_playing]:
                                        self.par_cross[self.now_playing].remove(l_x, l_y)

                    # add link3/4/5
                    # e.g. xxa, xax, xxxa, xxax, xxxax
                    if cnt + l_ext >= 4:
                        if (l_x, l_y) in self.link4[self.now_playing]:
                            self.link4[self.now_playing].remove((l_x, l_y))
                        self.link5[self.now_playing].add((l_x, l_y))
                    elif cnt + l_ext == 3:
                        if (l_x, l_y) in self.link3[self.now_playing]:
                            self.link3[self.now_playing].remove((l_x, l_y))
                        self.link4[self.now_playing].add((l_x, l_y))
                    elif cnt + l_ext == 2:
                        self.link3[self.now_playing].add((l_x, l_y))
                    if cnt + l_ext >= 2:
                        if (l_x, l_y) in self.dir[self.now_playing]:
                            if self.dir[self.now_playing][(l_x, l_y)] != (i, j):
                                self.cross[self.now_playing].add((l_x, l_y))
                        else:
                            self.dir[self.now_playing][(l_x, l_y)] = (i, j)
                    
                    # add partial link3/4
                    # e.g. xoax xaox, xxoa, xxoax, xxxoa
                    if l_ext == 0:
                        if l_par > 0:
                            par_dir_dict = self.par_dir[self.now_playing][(l_x, l_y)]
                            if cnt + l_par >= 3:
                                if (l_x, l_y) in self.par_link3:
                                    if par_dir_dict.get((i, j), 0) == 3:
                                        self.par_link3.remove((l_x, l_y))
                                self.par_link4[self.now_playing].add((l_x, l_y))
                                par_dir_dict[(i, j)] = 4
                            elif cnt + l_par == 2:
                                self.par_link3[self.now_playing].add((l_x, l_y))
                                par_dir_dict[(i, j)] = 3
                            else:
                                raise ValueError('cnt')
                            if len(par_dir_dict) + ((l_x, l_y) in self.dir[self.now_playing]) >= 2:
                                self.par_cross[self.now_playing].add((l_x, l_y))
                        if self.is_inside((ll_x, ll_y)) and self.board[ll_x][ll_y] == 0:
                            par_dir_dict = self.par_dir[self.now_playing][(ll_x, ll_y)]
                            if cnt + l_par >= 3:
                                if (ll_x, ll_y) in self.par_link3:
                                    if par_dir_dict.get((i, j), 0) == 3:
                                        self.par_link3.remove((ll_x, ll_y))
                                self.par_link4[self.now_playing].add((ll_x, ll_y))
                                par_dir_dict[(i, j)] = 4
                            elif cnt + l_par == 2:
                                self.par_link3[self.now_playing].add((ll_x, ll_y))
                                par_dir_dict[(i, j)] = 3
                            if len(par_dir_dict) + ((ll_x, ll_y) in self.dir[self.now_playing]) >= 2:
                                self.par_cross[self.now_playing].add((ll_x, ll_y))

                    del l_x, l_y, ll_x, ll_y, l_max, l_ext, l_par

                if self.is_inside((r_x, r_y)) and self.board[r_x][r_y] == 0:
                    for group in (self.par_link3, self.par_link4):
                        for player in (1, -1):
                            # remove same direction adjacant stone
                            if (r_x, r_y) in group and (i, j) in self.par_dir[player][(r_x, r_y)]:
                                group[player].remove((r_x, r_y))
                                self.par_dir[player][(r_x, r_y)].remove((i, j))
                    if cnt + r_ext >= 4:
                        if (r_x, r_y) in self.link4[self.now_playing]:
                            self.link4[self.now_playing].remove((r_x, r_y))
                        self.link5[self.now_playing].add((r_x, r_y))
                    elif cnt + r_ext == 3:
                        if (r_x, r_y) in self.link3[self.now_playing]:
                            self.link3[self.now_playing].remove((r_x, r_y))
                        self.link4[self.now_playing].add((r_x, r_y))
                    elif cnt + r_ext == 2:
                        self.link3[self.now_playing].add((r_x, r_y))
                    if cnt + r_ext >= 2:
                        if (r_x, r_y) in self.dir[self.now_playing]:
                            if self.dir[self.now_playing][(r_x, r_y)] != (i, j):
                                self.cross[self.now_playing].add((r_x, r_y))
                        else:
                            self.dir[self.now_playing][(r_x, r_y)] = (i, j)

                    if r_ext == 0:
                        if r_par > 0:
                            par_dir_dict = self.par_dir[self.now_playing][(r_x, r_y)]
                            if cnt + r_par >= 3:
                                if (r_x, r_y) in self.par_link3:
                                    if par_dir_dict.get((i, j), 0) == 3:
                                        self.par_link3.remove((r_x, r_y))
                                self.par_link4[self.now_playing].add((r_x, r_y))
                                par_dir_dict[(i, j)] = 4
                            elif cnt + r_par == 2:
                                self.par_link3[self.now_playing].add((r_x, r_y))
                                par_dir_dict[(i, j)] = 3
                            else:
                                raise ValueError('cnt')
                            if len(par_dir_dict) + ((r_x, r_y) in self.dir[self.now_playing]) >= 2:
                                self.par_cross[self.now_playing].add((r_x, r_y))
                        if self.is_inside((rr_x, rr_y)) and self.board[rr_x][rr_y] == 0:
                            par_dir_dict = self.par_dir[self.now_playing][(rr_x, rr_y)]
                            if cnt + r_par >= 3:
                                if (rr_x, rr_y) in self.par_link3:
                                    if par_dir_dict.get((i, j), 0) == 3:
                                        self.par_link3.remove((rr_x, rr_y))
                                self.par_link4[self.now_playing].add((rr_x, rr_y))
                                par_dir_dict[(i, j)] = 4
                            elif cnt + r_par == 2:
                                self.par_link3[self.now_playing].add((rr_x, rr_y))
                                par_dir_dict[(i, j)] = 3
                            if len(par_dir_dict) + ((rr_x, rr_y) in self.dir[self.now_playing]) >= 2:
                                self.par_cross[self.now_playing].add((rr_x, rr_y))


    def display_board(self) -> None:
        '''
        Print all placed stone.
        '''
        if self.moves == []:
            return
        else:
            i_ticks = '  0 1 2 3 4 5 6 7 8 9 A B C D E'
            i_ticks = i_ticks[0:1+2*self.size]
            print(i_ticks)
            for j in range(self.size):
                if j < 10:
                    print(j, end='')
                else:
                    print(chr(55 + j), end='')
                for i in range(self.size):
                    print(' ', end='')
                    if self.board[i][j] > 0:
                        print('o', end='')
                    elif self.board[i][j] < 0:
                        print('x', end='')
                    else:
                        print(' ', end='')
                    if i == self.size - 1:
                        print()
        return

    def adjacent_vacancies(self) -> Set[tuple]:
        '''
        ## Returns\n
        out: Set[tuple]
        A set which contains all available moves around existed stones. \
        'Around' means the horizontal AND vertival distance between a vacancy and \
        the nearest stone is no greater than 1.
        '''
        vacancies = set()
        if self.moves != []:
            bias = range(-1, 2)
            for move in self.moves:
                for i in bias:
                    if move[0]-i < 0 or move[0]-i >= self.size:
                        continue
                    for j in bias:
                        if move[1]-j < 0 or move[1]-j >= self.size:
                            continue
                        vacancies.add((move[0]-i, move[1]-j))

                if self.board[move[0]][move[1]] == self.now_playing:
                    for i in [-1, 1]:
                        if move[0]-i < 0 or move[0]-i >= self.size:
                            continue
                        for j in [-1, 1]:
                            if move[1]-j < 0 or move[1]-j >= self.size:
                                continue
                            vacancies.add((move[0]-i, move[1]-j))
                    for i, j in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                        if move[0]-i < 0 or move[0]-i >= self.size:
                            continue
                        if move[1]-j < 0 or move[1]-j >= self.size:
                            continue
                        vacancies.add((move[0]-i, move[1]-j))
            occupied = set(self.moves)
            vacancies -= occupied
        return vacancies

    def is_ended(self) -> bool:
        '''
        Judge whether the game is ended or not. The winner will be passed to `self.winner`. \
        The algorithm is not easy to understand. You can check it by traverse the `for` loop.
        ## Returns\n
        out: bool
            Return `True` if the game ended, otherwise `False`.
        '''
        if self.moves == []:
            return False
        loc_i, loc_j = self.moves[-1]
        color = -self.now_playing
        sgn_i = [1, 0, 1, 1]
        sgn_j = [0, 1, 1, -1]
        for iter in range(4):
            length = 0
            prm1 = loc_i if sgn_i[iter] == 1 else loc_j
            prm2 = loc_j if sgn_j[iter] == 1 else (loc_i if sgn_j[iter] == 0 else self.size - 1 - loc_j)
            start_bias = -min(prm1, prm2) if min(prm1, prm2) < self.win_len-1 else -self.win_len+1
            end_bias = self.size - 1 - max(prm1, prm2) if max(prm1, prm2) > self.size-self.win_len else self.win_len-1
            for k in range(start_bias, end_bias+1):
                stone = self.board[loc_i + k * sgn_i[iter]][loc_j + k * sgn_j[iter]]
                if color > 0 and stone > 0 or color < 0 and stone < 0:
                    length += 1
                else:
                    length = 0
                if length == self.win_len:
                    self.winner = 1 if color > 0 else -1
                    return True
        if len(self.moves) == self.size ** 2:
            return True
        else:
            return False