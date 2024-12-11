import torch
from torch import nn
# import torch_scatter
from torch_geometric.nn import MessagePassing
from constants import NodeType


hyperparams = {
    "n_layers": 2,
    "batch_size": 16,
    "hidden_dim": 10,
    "lr": 1e-3,
}

class ProcessingLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.node_mlp = nn.Sequential(
                nn.Linear(2*node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
                nn.LayerNorm(node_dim),
            )
        self.edge_mlp = nn.Sequential(
                nn.Linear(2*node_dim+edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim),
                nn.LayerNorm(edge_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for i in [0,2]:
            self.node_mlp[i].reset_parameters()
            self.edge_mlp[i].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Propagate messages to update embeddings.

        Args:
        x [n_nodes, node_dim]: node embeddings
        edge_index [2, n_edges]
        edge_attr [E, edge_dim]

        """
        # edge update
        out, updated_edges = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)

        # residual and node update
        updated_nodes = x + self.node_mlp(torch.cat([x, out], dim=1))
        return updated_nodes, updated_edges
    
    def message(self, x_i, x_j, edge_attr):
        """
        x_i [E, node_dim]: source node embedding 
        x_j [E, node_dim]: target node embedding 
        edge_attr [E, edge_dim]
        """
        updated_edges = torch.cat([x_i, x_j, edge_attr], dim=1)
        # edge update with residual
        return self.edge_mlp(updated_edges) + edge_attr

    def aggregate(self, updated_edges, edge_index, dim_size=None):
        # out = torch_scatter.scatter(updated_edges, edge_idx[0,:], dim=0, reduce="sum")
        # scatter add without dependency
        src = updated_edges
        out_size = list(src.size())
        out_size[0] = edge_index.max().item()+1
        out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
        out.index_add_(dim=0, index=edge_index[0, :], source=src)
        return out, updated_edges


class MeshGraphNet(nn.Module):
    def __init__(
            self,
            input_dim_node,
            input_dim_edge,
            output_dim,
            args,
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
            self.processor_blocks.append(ProcessingLayer(self.hidden_dim, self.hidden_dim))

        self.decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim)
                )

    def forward(self, data, mean_x, std_x, mean_edge, std_edge):
        x, edge_idx, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p

        # normalize nodes and edges
        x = (x-mean_x)/std_x
        edge_attr = (edge_attr-mean_edge)/std_edge


        # encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # update embeddings using processor layers
        for layer in self.processor_blocks:
            x, edge_attr = layer(x, edge_idx, edge_attr)

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


        
    
