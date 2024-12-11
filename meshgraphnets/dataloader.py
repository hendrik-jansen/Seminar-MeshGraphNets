import torch
from torch_geometric.data import Data
from tfrecord.torch.dataset import TFRecordDataset
import os
import json

datapath = os.path.join(os.path.dirname(__file__), "..", "data", "flag_simple") 

with open(os.path.join(datapath, "meta.json"), 'r') as file:
    meta = json.load(file)

description = {}
for k,v in meta["features"].items():
    dtype = v["dtype"]
    dtype = "byte" if dtype.startswith("int") else dtype
    description[k] = dtype


trainset = TFRecordDataset(os.path.join(datapath, "test.tfrecord"), index_path=None, description=description)
loader = torch.utils.data.DataLoader(trainset)
