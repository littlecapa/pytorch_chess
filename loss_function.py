import torch
import sys
import logging
from chess_nn_trainer_stats import Chess_NN_Trainer_Stats

basic_eval_classes = ['equal', 'slight_plus', 'clear_plus', 'decisive', 'clear_win']
basic_eval_boundaries = [0.2, 1.0, 2.0, 4.0, sys.maxsize]
basic_eval_values = [0, 0.5, 1, 2, 4]

def get_eval_classes(eval):
    for index, b in enumerate(basic_eval_boundaries):
        if eval < b:
            return index
        
def same_sign_or_zero(tensor1, tensor2):
    # Compute the signs of the tensors
    sign1 = torch.sign(tensor1)
    sign2 = torch.sign(tensor2)

    # Check if both tensors have the same sign or either is 0.0
    result = (sign1 == sign2) | (tensor1 == 0.0) | (tensor2 == 0.0)
    return result

def compute_loss_tensor(outputs_classes, labels_classes, delta_tensor, same_sign, device):
    delta_classes = torch.abs(outputs_classes - labels_classes).to(device)
    delta_classes_values = torch.gather(torch.tensor(basic_eval_values).to(device), 0, delta_classes).to(device)
    output_values = torch.gather(torch.tensor(basic_eval_values).to(device), 0, outputs_classes).to(device)
    label_values = torch.gather(torch.tensor(basic_eval_values).to(device), 0, labels_classes).to(device)
    classes_values = torch.where(same_sign,
                                delta_classes_values,
                                outputs_classes + labels_classes).to(device)
    delta_values = torch.where(same_sign,
                               (delta_tensor * 1e-2) * (delta_classes + 1),
                               (delta_tensor * 1e-2) * ((outputs_classes + labels_classes)).to(device))
    return classes_values + delta_values

def eval_loss_function(outputs, labels, device, stats):
    same_sign = same_sign_or_zero(outputs, labels).to(device)
    delta_tensor = torch.abs(outputs - labels).to(device)
    outputs_classes = torch.tensor([get_eval_classes(abs(value)) for value in outputs]).to(device)
    labels_classes = torch.tensor([get_eval_classes(abs(value)) for value in labels]).to(device)
    loss_tensor = compute_loss_tensor(outputs_classes, labels_classes, delta_tensor, same_sign, device)
    stats.write_loss_function_infos(outputs, labels, loss_tensor)
    return torch.mean(loss_tensor)
    
