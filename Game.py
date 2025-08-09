import chess
import torch
import copy
import math
import numpy as np
from Net import NeuralNetwork, convert_bb

print("Finished imports (this is Game.py)")

nn = NeuralNetwork()
nn.load_state_dict(torch.load("/Users/akshith/PycharmProjects/pythonProject49/model.pth"))

print("Successfully loaded model")


class Node:
    def __init__(self, board, par, turn):
        self.board = board
        self.par = par
        self.wins = 0
        self.visits = 0
        self.exploration_parameter = 2 ** 0.5
        self.turn = turn
        self.children = []

    def get_best_child(self):
        return max(self.children, key=lambda c: c.wins / c.visits + self.exploration_parameter * (math.log(self.visits) / c.visits) ** 0.5)

    def expand(self):
        for next_move in self.board.legal_moves:
            self.board.push(next_move)
            if all(child.board != self.board for child in self.children):
                self.children.append(Node(copy.deepcopy(self.board), self, turn=1 - self.turn))
                self.board.pop()
                return True
            self.board.pop()
        return False

    def playout(self):
        copy_board = copy.deepcopy(self.board)
        while not copy_board.is_game_over() and copy_board.fullmove_number < 50:
            move = np.random.choice(list(copy_board.legal_moves))
            copy_board.push(move)
        if not copy_board.is_game_over() or copy_board.result() == "1/2-1/2":
            bbs = []
            for colour in (chess.WHITE, chess.BLACK):
                for piece in (chess.PAWN, chess.ROOK, chess.QUEEN, chess.KING, chess.KNIGHT, chess.BISHOP):
                    bb = list(copy_board.pieces(piece, colour))
                    bbs.append(convert_bb(bb))
            bbs = np.array(bbs, dtype=np.float32)
            bbs = np.ndarray.flatten(bbs)
            result = nn.forward(torch.tensor(np.expand_dims(bbs, axis=0), dtype=torch.float32))[0] > 0.5
            return result
        else:
            return 1 if copy_board.result() == "1-0" else 0 if copy_board.result() == "0-1" else None

    def backpropagate(self, result):
        self.wins += int(result == self.turn)
        self.visits += 1
        if self.par is not None:
            self.par.backpropagate(result)


print("Starting game!")
board = chess.Board()
while not board.is_game_over():
    white = Node(board, None, 1)
    next = copy.deepcopy(white)
    for _ in range(500):
        while next.children and not next.board.is_game_over():
            next = next.get_best_child()
        while not next.expand():
            next = next.par
        next = next.children[-1]
        result = next.playout()
        next.backpropagate(result)
        while next.par is not None:
            next = next.par
    best = next.get_best_child().board.fen()
    board.set_fen(best)

    black = Node(board, None, 0)
    next = copy.deepcopy(black)
    for _ in range(500):
        while next.children and not next.board.is_game_over():
            next = next.get_best_child()
        while not next.expand():
            next = next.par
        next = next.children[-1]
        result = next.playout()
        next.backpropagate(result)
        while next.par is not None:
            next = next.par
    best = next.get_best_child().board.fen()
    board.set_fen(best)

    print(board)

