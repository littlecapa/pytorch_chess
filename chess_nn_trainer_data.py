import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from chess_eval import Chess_Eval
from tensor_lib import convert_int64_to_int32


class Chess_NN_Trainer_Data(Dataset):

    def __init__(self, data_path, scale_eval = True):
        self.data = []
        self.labels = []
        self.scale_eval = scale_eval
        self.eval = Chess_Eval()
        self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.scale_eval:
            logging.debug(f"X64: {self.data[idx]}")
            return convert_int64_to_int32(self.data[idx]), self.eval.align_eval(self.labels[idx])
        return convert_int64_to_int32(self.data[idx]), self.labels[idx]
    
    def load_data(self, data_path):
        with open(data_path, 'r') as file:
            lines = file.readlines()
        logging.debug(f"File: {data_path}, Lines: {len(lines)}")
        for line in lines:
            values = line.strip().split(';')
            # -2 wegen ; nach letztem Element, sonst wäre das -1
            feature_values_int = [int(value) for value in values[:-2]]
            #Convert to 2xint32/float32/ => get item
            label_value = float(values[-2])
            #Alternative: Count Material
            self.data.append(torch.tensor(feature_values_int, dtype=torch.int64))
            self.labels.append(label_value)
