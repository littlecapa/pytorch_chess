import torch
import torch.nn as nn
import logging

from chess_halfkp import Chess_HalfKP

class Chess_HalfKP_Binary(Chess_HalfKP):
#
# Define a PyTorch net that resembles the Stockfish HalfKP net
#
    def __init__(self):
        super(Chess_HalfKP_Binary, self).__init__()
        self.init(13*64)