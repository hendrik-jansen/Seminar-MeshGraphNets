# MeshGraphNets: Learning Mesh-based Simulation with Graph Networks

## I. Introduction
Finite element simulations are a standard for modeling mesh systems, but they are expensive and often domain specific.
Graph neural networks can learn the underlying dynamics in a domain agnostic way and with much cheaper inference.

## II. Background

### II.A. Traditional Simulation Methods
Conventional partial differential equation solvers iteratively compute the system’s state but scale quickly in computational cost with mesh size and resolution.

### II.B. Graph Neural Networks
GNNs can learn local interactions by learning node and edge embeddings and propagating this signal along edges. 

### II.C. Adaptive Remeshing
Depending on local complexity, different regions of the mesh require different resolutions to balance accuracy and computational cost.
Adaptive meshing refines or coarsens regions based on different criteria expressed through a sizing field. Traditionally this requires domain knowledge, but the sizing field can also be learned to loosen this restriction. (Implementation not published)

## III. Domains & Datasets
The authors evaluate on cloth simulation, structural deformation, and both compressible and incompressible fluid dynamics. The training data is generated from different simulators.

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
In addition the model is remeshed as described in II.C
The authors work with the TFrecord data format, we use Rayan Kanfar's (https://github.com/kanfarrs/) translation code to obtain torch compatible data.
Dataset statistics (mean + std dev) are collected from train data and applied to normalize at inference.
The model operates on relative coordinates for robustness.


## IV. Model Architecture
The chosen architecture is very similar to unnormalized graph convolution, with the addition of edge embeddings.
First both nodes and edges are embedded, then in a number of processing steps the embeddings are updated according to adjacency.
Finally the node embeddings are decoded.


## V. Results
MeshGraphNets deliver accurate long rollouts at speeds one to two orders of magnitude faster than the solvers. They outperform particle-based and grid-based baselines, with smaller errors in challenging regimes. They also scale and generalize well at inference, enabling simulations larger than or fundamentally different from any seen in training.
# TODO include tables and ablation
