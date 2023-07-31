import logging
import torch
from torch.utils.data import Dataset
from tensor_lib import convert_int64_to_int32

class Chess_NN_Trainer_Test_Data(Dataset):

    def __init__(self, data_path):
        self.positions = []
        self.evaluations = []
        self.infos = []
        self.load_data(data_path)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return convert_int64_to_int32(self.positions[idx]), torch.tensor(float(self.evaluations[idx]))
    
    def get_test_ingo(self, idx):
        return self.infos[idx]
        
    def load_data(self, data_path):
        with open(data_path, 'r') as file:
            lines = file.readlines()
        logging.info(f"File: {data_path}, Lines: {len(lines)}")
        for line in lines:
            values = line.strip().split(';')
            # -2 wegen ; nach letztem Element, sonst wäre das -1
            position = [int(value) for value in values[:-2]]
            evaluation = values[-2]
            info = values[-1]
            #Alternative: Count Material
            self.positions.append(torch.tensor(position, dtype=torch.int64))
            self.evaluations.append(evaluation)
            self.infos.append(info)