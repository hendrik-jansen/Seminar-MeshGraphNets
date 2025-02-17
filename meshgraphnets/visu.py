import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np
from load_preprocessed import load_dataset

# Extract static data once
data_list = load_dataset()
data = data_list[0]
x = data.mesh_pos.numpy()
edge_index = data.edge_index.numpy()
N_FRAMES = 50

# Prepare the plot once
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(x[:, 0], x[:, 1], c=data_list[0].p.numpy().flatten(), cmap='viridis', s=5)
for i in range(edge_index.shape[1]):
    src, dest = edge_index[:, i]
    ax.plot([x[src, 0], x[dest, 0]], [x[src, 1], x[dest, 1]], color='gray', lw=0.1)
ax.axis('equal')
ax.axis('off')


pressures_diffs = []
for i in range(N_FRAMES):
    pressures_diffs.append(data_list[i].p.numpy())

pressures = np.cumsum(pressures_diffs, axis=0)



# Function to update the colors
def update(frame):
    scatter.set_array(pressures[frame].flatten())
    ax.set_title(f"Frame {frame + 1}")

# Create animation
ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100)

# Save as GIF
ani.save("animation.gif", writer="pillow", fps=4)

