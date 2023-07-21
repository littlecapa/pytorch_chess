import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from chess_eval import Chess_Eval


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
            return self.convert_int64_to_int32(self.data[idx]), self.eval.align_eval(self.labels[idx])
        return self.convert_int64_to_int32(self.data[idx]), self.labels[idx]

    def convert_int64_to_int32(self, int64_tensor):
        # Check if the input tensor has the correct shape (13x1)
        if int64_tensor.shape != (13, 1):
            raise ValueError("Input tensor must have shape (13, 1)")

        # Convert the int64 tensor to a list of int64 values
        int64_values = int64_tensor.squeeze().tolist()

        # Initialize an empty list to store the int32 values
        int32_values = []

        # Convert each int64 value to two int32 values
        for int64_value in int64_values:
            # Convert the int64 value to 8 bytes (64 bits)
            bytes_64 = int64_value.to_bytes(8, byteorder='big', signed=True)
            # Convert the 4-byte chunks to int32 values
            int32_value_1 = int.from_bytes(bytes_64[:4], byteorder='big', signed=True)
            int32_value_2 = int.from_bytes(bytes_64[4:], byteorder='big', signed=True)

            # Append the int32 values to the list
            int32_values.append(int32_value_1)
            int32_values.append(int32_value_2)

        # Convert the list of int32 values to a 26x1 int32 tensor
        int32_tensor = torch.tensor(int32_values, dtype=torch.int32).unsqueeze(1)

        return int32_tensor

    
    def load_data(self, data_path):
        with open(data_path, 'r') as file:
            lines = file.readlines()
        logging.info(f"File: {data_path}, Lines: {len(lines)}")
        for line in lines:
            values = line.strip().split(';')
            # -2 wegen ; nach letztem Element, sonst wäre das -1
            feature_values_int = [int(value) for value in values[:-2]]
            #To Do: Convert to 2xint32/float32/..
            label_value = float(values[-2])
            #Alternative: Count Material
            self.data.append(torch.tensor(feature_values_int, dtype=torch.int64))
            self.labels.append(label_value)
