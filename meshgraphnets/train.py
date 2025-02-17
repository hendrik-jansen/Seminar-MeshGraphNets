import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from tqdm import trange

from model import MeshGraphNet
from alt_passing import AltMeshGraphNet
from load_preprocessed import load_testset
from normalization import get_stats



TRAIN_SIZE = 10
TEST_SIZE = 2
DEVICE = "cpu"

ARGS = {
    "num_layers": 10,
    "batch_size": 16,
    "hidden_dim": 10,
    "epochs": 10,
    "lr": 1e-3,
    "weight_decay": 5e-4,
}

def train():
    dataset = load_testset()
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

    epoch_losses = []

    for epoch in trange(ARGS["epochs"]):
        model.train()
        batch_losses = []
        for batch in dataloader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch, mean_x, std_x, mean_edge, std_edge)
            loss = model.loss(pred, batch, mean_y, std_y)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        
        # Calculate the average loss for the epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        print(avg_loss)
        epoch_losses.append(avg_loss)

    # Plot the average losses
    plt.plot(range(1, ARGS["epochs"] + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Epoch')
    plt.show()



if __name__ == "__main__":
    train()
