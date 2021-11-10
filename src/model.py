import torch
from torch_geometric.nn import GAE, VGAE
from encoder import *
from os import environ as env

avaiable_models = {
    "GAE" : GAE,
    "VGAE" : VGAE
    }

def MODEL(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = ENCODER(config["ENCODER"])
    decoder = None
    model = avaiable_models[config["TYPE"]](encoder, None)
    model = model.to(device)
    return model