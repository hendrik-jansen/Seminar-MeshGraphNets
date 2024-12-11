import torch
import os

DATAPATH = os.path.abspath(os.path.dirname(__file__)) + "/../data/"
def load_dataset():
    return torch.load(DATAPATH + "meshgraphnets_miniset5traj_vis.pt", weights_only=False)

def load_testset():
    return torch.load(DATAPATH + "test_processed_set.pt", weights_only=False)

