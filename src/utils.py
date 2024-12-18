# This script contains utility functions for weight initialization and aggregation.

import torch

def initialize_weights(model):
    return {key: torch.zeros_like(param) for key, param in model.state_dict().items()}

def aggregate_weights(weights_list):
    aggregated_weights = {key: torch.zeros_like(weights_list[0][key]) for key in weights_list[0]}
    for key in aggregated_weights.keys():
        for weights in weights_list:
            aggregated_weights[key] += weights[key]
        aggregated_weights[key] /= len(weights_list)
    return aggregated_weights
