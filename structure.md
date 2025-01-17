# MeshGraphNets: Learning Mesh-based Simulation with Graph Networks

## I. Introduction
Finite element simulations are a standard for modeling mesh systems, but they are expensive and often domain specific.
Graph neural networks can learn the underlying dynamics in a domain agnostic way and with much cheaper inference.

## II. Background

### II.A. Traditional Simulation Methods
Conventional partial differential equation solvers iteratively compute the system’s state but scale quickly in computational cost with mesh size and resolution.
They are however fairly accurate, interpretable and stable even on long rollouts, as they directly encode the physical laws underlying the modeled interactions.

### II.B. Graph Neural Networks
Graph-structured data presents challenges to common machine learning approaches. Naively processing an adjacency matrix with MLPs discards much of the structure, does not generalize to graphs of different sizes and quickly becomes computationally intractable.
Instead the paradigm of message-passing has emerged, wherein each node is embedded individually and then updated by aggregation of embeddings from adjacent nodes.
The concrete embedding and aggregation scheme is a degree of freedom and several architectures have been proposed, including siblings of non-geometric ML ideas, such as:
- Graph Convolution: sum of neighbors, weighted by the root of the degree of both nodes.
- Graph Attention: learnable weighted sum of adjacent embeddings.
Since each GNN block processes a single hop, the receptive field at each node is simply the number of layers.
Typically this is limited, forcing the network to learn local dynamics, which helps in generalization and prevents oversmoothing of predictions.

### II.C. Adaptive Remeshing
Depending on local complexity, different regions of the mesh require different resolutions to balance accuracy and computational cost.
Adaptive meshing refines or coarsens regions based on different criteria expressed through a sizing field. Traditionally this requires domain knowledge, but the sizing field can also be learned to loosen this restriction. (Implementation not published)

## III. Domains & Datasets
The authors evaluate on cloth simulation, structural deformation, and both compressible and incompressible fluid dynamics. The training data is generated from different simulators.
The method capable of handling different domains with minimal, if any, adaptation. Many of the problems application even use the same hyperparameters.

| Dataset        | System      | Solver | Mesh Type   | Meshing      | # Steps | ∆t/s |
|----------------|-------------|--------|-------------|--------------|---------|-------|
| FLAGSIMPLE     | cloth       | ArcSim | triangle    | 3D regular   | 400     | 0.02  |
| FLAGDYNAMIC    | cloth       | ArcSim | triangle    | 3D dynamic   | 250     | 0.02  |
| SPHEREDYNAMIC  | cloth       | ArcSim | triangle    | 3D dynamic   | 500     | 0.01  |
| DEFORMINGPLATE | hyper-el.   | COMSOL | tetrahedral | 3D irregular | 400     | —     |
| CYLINDERFLOW   | incompr. NS | COMSOL | triangle    | 2D irregular | 600     | 0.01  |
| AIRFOIL        | compr. NS   | SU2    | triangle    | 2D irregular | 600     | 0.008 |

This notebook uses the example of cylinderflow.
The specific dataset is a timeseries $[g_0, ... g_n]$
Where each $g_i$ consists of:
- x (node features): 2D velocity + one hot encodeded node type. Shape: [n_nodes, 11]
- edge index: adjacency. Shape: [2, n_edges]
- edge attributes. 2D position + 2-norm. Shape: [n_edges, 3]
- y (node outputs). Difference in fluid acceleration to next step (think residual prediction). Shape: [n_nodes, 2]
- p (pressure): Unused. Shape: [n_nodes]

### III.A. Preprocessing
The mesh is augmented with world edges, nodes that are close in world space are connected to model collisions.
Originally two separate edge MLPs are learned for this purpose, however indicating whether an edge belongs to the mesh- or world graph as part of the edge features is equally as expressive and this step therefore ignored in our reimplementation.
In addition the graph is remeshed as described in II.C
The authors (deepmind) work with the tensorflow TFrecord data format, we use Rayan Kanfar's (https://github.com/kanfarrs/) translation code to obtain torch compatible data.
Dataset statistics (mean + std dev) are collected from train data and applied to normalize at inference.
The model operates on relative coordinates for robustness.


## IV. Model Architecture
The chosen architecture is very similar to unnormalized graph convolution, with the addition of edge embeddings.
First nodes and edges are embedded, then in a number of processing steps the embeddings are updated according to adjacency.
At each step the edge embeddings are updated via MLP of concatenated adjacent nodes + residual.
Then the node embeddings are updated via MLP of previous node embedding concatenated with the edge embeddings + residual.
Finally the node embeddings are decoded for the quantity of interest (f.e. velocity).


## V. Results
MeshGraphNets deliver accurate long rollouts at speeds up to two orders of magnitude faster than the solvers. They outperform particle-based and grid-based baselines, with smaller errors in challenging regimes. They also scale and generalize well at inference, enabling simulations larger than or fundamentally different from anything seen in training.
# TODO include tables and ablation

## VI. Tricks

### VI.A Training Noise
To improve rollout stability, noise can be injected during training, nudging the model towards self correcting behavior.
As small predictions inevitably accumulate at inference, the authors report that a model used to noisy environments is much more robust to this.
Empirically predictions stay plausible even after tens of thousands of rollout steps.
