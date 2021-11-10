import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

available_datasets = {"Planetoid" : Planetoid}

def DATASET(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=config["VALSIZE"], num_test=config["TESTSIZE"], is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', config["FAMILY"])
    data = available_datasets[config["FAMILY"]](path, config["NAME"], transform=transform)
    return data