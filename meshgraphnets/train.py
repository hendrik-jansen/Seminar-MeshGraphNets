import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from model import MeshGraphNet
from load_preprocessed import load_dataset
from normalization import get_stats



TRAIN_SIZE = 10
TEST_SIZE = 2
DEVICE = "cpu"

ARGS = {
    "num_layers": 5,
    "batch_size": 8,
    "hidden_dim": 10,
    "epochs": 10,
    "lr": 1e-3,
    "weight_decay": 5e-4,
}

def train():
    dataset = load_dataset()
    dataloader = DataLoader(dataset[:TRAIN_SIZE])
    testloader = DataLoader(dataset[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE])

    [mean_x,std_x,mean_edge,std_edge,mean_y,std_y] = list(map(lambda x: x.to(DEVICE), get_stats(dataset)))

    model = MeshGraphNet(
            input_dim_node=dataset[0].x.shape[1],
            input_dim_edge=dataset[0].edge_attr.shape[1],
            output_dim=2, # velocity
            args=ARGS,
            )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(trainable_params, lr=ARGS["lr"], weight_decay=ARGS["weight_decay"])

    for epoch in tqdm(range(ARGS["epochs"])):
        model.train()
        for batch in dataloader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch, mean_x, std_x, mean_edge, std_edge)
            loss = model.loss(pred, batch, mean_y, std_y)
            loss.backward()
            opt.step()
            print(loss.item())



if __name__ == "__main__":
    train()
