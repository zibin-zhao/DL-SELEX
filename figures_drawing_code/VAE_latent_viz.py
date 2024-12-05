'''Drawing latent space for AptaClux by Zibin Zhao'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Create the custom colormap using two colors
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#d98380'])

# Create the figure and a 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate data for the surface plot with three distinct peaks
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Create a surface with three distinct peaks of different heights
z = (
    1.0 * np.exp(-0.2 * ((x + 3) ** 2 + y ** 2))  # First peak (height 1.0)
    + 0.6 * np.exp(-0.2 * ((x - 3) ** 2 + y ** 2))  # Second peak (height 0.6)
    + 0.4 * np.exp(-0.2 * (x ** 2 + (y - 3) ** 2))  # Third peak (height 0.4)
)

# Plot the surface with the custom colormap
surf = ax.plot_surface(x, y, z, cmap=custom_cmap, edgecolor='none', alpha=0.8)

# Add a contour plot on the bottom (to represent the bottom plane)
ax.contourf(x, y, z, zdir='z', offset=-0.2, cmap=custom_cmap, alpha=0.4)

# Marking some points to indicate the "Next location" movement
points_x = [-3, 0, 3]
points_y = [0, -3, 0]
points_z = [
    1.0 * np.exp(-0.2 * ((-3) ** 2 + 0 ** 2)),  # First peak point
    0.4 * np.exp(-0.2 * ((0) ** 2 + (-3) ** 2)),  # Third peak point
    0.6 * np.exp(-0.2 * ((3) ** 2 + 0 ** 2)),  # Second peak point
]

# Adding a path line to show movement (dotted)
ax.plot(points_x, points_y, points_z, 'w--', linewidth=2)

# Centering the graph by adjusting axis limits
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-0.2, 1.0])

# Hide the axes and ticks
ax.set_xticks([])  # Hide X ticks
ax.set_yticks([])  # Hide Y ticks
ax.set_zticks([])  # Hide Z ticks

# Remove axis spines and labels for a minimalistic look
ax.set_axis_off()

# Adjust the view angle for a more professional look
ax.view_init(30, 130)

# Tight layout for professional spacing
plt.tight_layout()

# Save the plot with high DPI for publication (Nature-style)
plt.savefig('vae_viz.tiff', dpi=300, format='tiff', facecolor='#FFFFFF')

# Show the plot
plt.show()
