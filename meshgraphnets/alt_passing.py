import torch
from torch import nn
# import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN, GAT
from constants import NodeType


hyperparams = {
    "n_layers": 2,
    "batch_size": 16,
    "hidden_dim": 10,
    "lr": 1e-3,
}


class AltMeshGraphNet(nn.Module):
    def __init__(
            self,
            input_dim_node,
            input_dim_edge,
            output_dim,
            args,
            agg_scheme="GCN",
        ):
        super().__init__()
        self.num_layers = args["num_layers"]
        self.hidden_dim = args["hidden_dim"]
        self.node_encoder = nn.Sequential(
                nn.Linear(input_dim_node, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )

        self.edge_encoder = nn.Sequential(
                nn.Linear(input_dim_edge, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )

        self.processor_blocks = []
        for _ in range(self.num_layers):
            if agg_scheme == "GCN":
                layer = GCN(in_channels=self.hidden_dim, hidden_channels=self.hidden_dim, num_layers=1, out_channels=self.hidden_dim)
            elif agg_scheme == "GAT":
                layer = GAT(in_channels=self.hidden_dim, hidden_channels=self.hidden_dim, num_layers=1, out_channels=self.hidden_dim)
            else:
                raise NotImplementedError
            self.processor_blocks.append(layer)

        self.decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim)
                )

    def forward(self, data, mean_x, std_x, mean_edge, std_edge):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # normalize nodes and edges
        x = (x-mean_x)/std_x
        edge_attr = (edge_attr-mean_edge)/std_edge


        # encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # update embeddings using processor layers
        for layer in self.processor_blocks:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # decode node embeddings
        return self.decoder(x)

    def loss(self, pred, inputs, mean_y, std_y):
        # normalize labels
        labels = (inputs.y-mean_y)/std_y

        # create mask for nodes of interest
        loss_mask = torch.logical_or((torch.argmax(inputs.x[:,2:],dim=1)==NodeType.NORMAL),
                                    (torch.argmax(inputs.x[:,2:],dim=1)==NodeType.OUTFLOW))

        # calculate total error 
        loss = torch.sum((labels-pred)**2, axis=1)
        # mask and mean
        return torch.mean(loss[loss_mask])


   
