"""Vendored Gridworld from *Deep Reinforcement Learning in Action*.

Original source: https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction
License: MIT (see upstream repository).

Files merged into a single module:
- Environments/GridBoard.py
- Environments/Gridworld.py
"""
from __future__ import annotations

import numpy as np


def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


class BoardPiece:
    def __init__(self, name, code, pos):
        self.name = name
        self.code = code
        self.pos = pos


class BoardMask:
    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self):
        return np.nonzero(self.mask)


def zip_positions2d(positions):
    x, y = positions
    return list(zip(x, y))


def addTuple(a, b):
    return tuple(sum(x) for x in zip(a, b))


class GridBoard:
    def __init__(self, size=4):
        self.size = size
        self.components: dict[str, BoardPiece] = {}
        self.masks: dict[str, BoardMask] = {}

    def addPiece(self, name, code, pos=(0, 0)):
        self.components[name] = BoardPiece(name, code, pos)

    def addMask(self, name, mask, code):
        self.masks[name] = BoardMask(name, mask, code)

    def movePiece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def render(self):
        dtype = "<U2"
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = " "
        for _, piece in self.components.items():
            displ_board[piece.pos] = piece.code
        for _, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code
        return displ_board

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for _, piece in self.components.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1
        for _, _mask in self.masks.items():
            x, y = self.masks["boundary"].get_positions()
            z = np.repeat(layer, len(x))
            displ_board[(z, x, y)] = 1
            layer += 1
        return displ_board


class Gridworld:
    def __init__(self, size=4, mode="static"):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            self.board = GridBoard(size=4)

        self.board.addPiece("Player", "P", (0, 0))
        self.board.addPiece("Goal", "+", (1, 0))
        self.board.addPiece("Pit", "-", (2, 0))
        self.board.addPiece("Wall", "W", (3, 0))

        if mode == "static":
            self.initGridStatic()
        elif mode == "player":
            self.initGridPlayer()
        else:
            self.initGridRand()

    def initGridStatic(self):
        self.board.components["Player"].pos = (0, 3)
        self.board.components["Goal"].pos = (0, 0)
        self.board.components["Pit"].pos = (0, 1)
        self.board.components["Wall"].pos = (1, 1)

    def validateBoard(self):
        player = self.board.components["Player"]
        goal = self.board.components["Goal"]
        wall = self.board.components["Wall"]
        pit = self.board.components["Pit"]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False
        corners = [
            (0, 0),
            (0, self.board.size),
            (self.board.size, 0),
            (self.board.size, self.board.size),
        ]
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [
                self.validateMove("Player", addpos)
                for addpos in [(0, 1), (1, 0), (-1, 0), (0, -1)]
            ]
            val_move_go = [
                self.validateMove("Goal", addpos)
                for addpos in [(0, 1), (1, 0), (-1, 0), (0, -1)]
            ]
            if 0 not in val_move_pl or 0 not in val_move_go:
                return False
        return True

    def initGridPlayer(self):
        self.initGridStatic()
        self.board.components["Player"].pos = randPair(0, self.board.size)
        if not self.validateBoard():
            self.initGridPlayer()

    def initGridRand(self):
        self.board.components["Player"].pos = randPair(0, self.board.size)
        self.board.components["Goal"].pos = randPair(0, self.board.size)
        self.board.components["Pit"].pos = randPair(0, self.board.size)
        self.board.components["Wall"].pos = randPair(0, self.board.size)
        if not self.validateBoard():
            self.initGridRand()

    def validateMove(self, piece, addpos=(0, 0)):
        outcome = 0
        pit = self.board.components["Pit"].pos
        wall = self.board.components["Wall"].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1
        elif max(new_pos) > (self.board.size - 1):
            outcome = 1
        elif min(new_pos) < 0:
            outcome = 1
        elif new_pos == pit:
            outcome = 2
        return outcome

    def makeMove(self, action):
        def checkMove(addpos):
            if self.validateMove("Player", addpos) in [0, 2]:
                new_pos = addTuple(self.board.components["Player"].pos, addpos)
                self.board.movePiece("Player", new_pos)

        if action == "u":
            checkMove((-1, 0))
        elif action == "d":
            checkMove((1, 0))
        elif action == "l":
            checkMove((0, -1))
        elif action == "r":
            checkMove((0, 1))

    def reward(self):
        if self.board.components["Player"].pos == self.board.components["Pit"].pos:
            return -1
        elif self.board.components["Goal"].pos == self.board.components["Player"].pos:
            return 1
        else:
            return 0

    def display(self):
        return self.board.render()
