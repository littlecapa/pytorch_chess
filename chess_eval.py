import logging
import chess
from enum import Enum
import numpy as np

class Chess_Eval():
  MIN = -1500
  MAX = 1500

  def align_eval(self, float_eval):
    value = round(float_eval*100)
    return min(max(value, self.MIN), self.MAX)/100

  piece_values = {
            chess.PAWN: 1.,
            chess.KNIGHT: 3.,
            chess.BISHOP: 3.5,
            chess.ROOK: 5.,
            chess.QUEEN: 9.
        }

  class pieces(Enum):
            white_pawns = 0
            white_knights = 1
            white_bishops = 2
            white_rooks = 3
            white_queens = 4
            white_king = 5
            black_pawns = 6
            black_knights = 7
            black_bishops = 8
            black_rooks = 9
            black_queens = 10
            black_king = 11

  def count_ones(self, bit_array):
    return sum(1 for bit in bit_array if bit)

  def count_material(self, bit_board):
    score = 0
    for i in range(12):
      ones = self.count_ones(bit_board[i])
      if ones > 0:
        if i == self.pieces.white_pawns.value:
          score += ones * self.piece_values[chess.PAWN]
        elif i == self.pieces.black_pawns.value:
          score -= ones * self.piece_values[chess.PAWN]
        elif i == self.pieces.white_knights.value:
          score += ones * self.piece_values[chess.KNIGHT]
        elif i == self.pieces.black_knights.value:
          score -= ones * self.piece_values[chess.KNIGHT]
        elif i == self.pieces.white_bishops.value:
          score += ones * self.piece_values[chess.BISHOP]
        elif i == self.pieces.black_bishops.value:
          score -= ones * self.piece_values[chess.BISHOP]
        elif i == self.pieces.white_rooks.value:
          score += ones * self.piece_values[chess.ROOK]
        elif i == self.pieces.black_rooks.value:
          score -= ones * self.piece_values[chess.ROOK]
        elif i == self.pieces.white_queens.value:
          score += ones * self.piece_values[chess.QUEEN]
        elif i == self.pieces.black_queens.value:
          score -= ones * self.piece_values[chess.QUEEN]
    return self.align_eval(score)