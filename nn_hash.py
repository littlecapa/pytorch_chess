import torch.nn as nn
import hashlib

class nn_Hash(nn.Module):

  def get_hash_value(self):
    # Serialize the model's state_dict into a string
    state_dict_str = str(self.state_dict()) + str(self.named_parameters)
    # Compute the SHA-256 hash of the state_dict string
    sha256_hash = hashlib.sha256(state_dict_str.encode()).hexdigest()
    # Convert the hexadecimal hash to an integer
    hash_value = abs(int(sha256_hash, 16))
    return hash_value