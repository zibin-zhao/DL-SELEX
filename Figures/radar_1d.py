import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Define the labels for the radar chart (6-label configuration)
labels = ['F1', 'Recall', 'Specificity', 'Accuracy', 'NPV', 'PPV']
num_vars = len(labels)

# Updated values for each model (you can replace this with your new model data)
values_1D = [0.68, 0.70, 0.62, 0.84, 0.76, 0.64]
values_1D_no_MF = [0.72, 0.66, 0.54, 0.74, 0.73, 0.76]
values_2D = [0.76, 0.74, 0.78, 0.77, 0.76, 0.75]
values_2D_no_MF = [0.70, 0.68, 0.73, 0.72, 0.71, 0.69]
values_3D = [0.82, 0.84, 0.86, 0.88, 0.82, 0.84]
values_3D_no_MF = [0.74, 0.72, 0.77, 0.75, 0.74, 0.73]
values_1D_3D = [0.92, 0.86, 0.86, 0.85, 0.88, 0.87]

# Create the angles for the radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # To close the circle

# Append the first value to the end to close the shape for each dataset
values_1D += values_1D[:1]
values_1D_no_MF += values_1D_no_MF[:1]
values_2D += values_2D[:1]
values_2D_no_MF += values_2D_no_MF[:1]
values_3D += values_3D[:1]
values_3D_no_MF += values_3D_no_MF[:1]
values_1D_3D += values_1D_3D[:1]

# Set the font to Arial and background to white
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

# Define the color codes for each configuration
colors = ['#8E0F31',  # Dark Red
          '#024163',  # Dark Blue
          '#066190',  # Medium Blue
          '#77AECD',  # Light Blue
          '#B57979',  # Dusty Red
          '#576FA0',  # Cool Blue
          '#f9c00f']  # Orange for 1D + 3D

# Define the markers for each configuration
markers = ['o', 's', 'D', '^', '*', 'p', 'v']  # Circle, Square, Diamond, Triangle, Star, Plus, Inverted Triangle

# Initialize the radar chart with a square aspect ratio
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# Plot each configuration with transparency, filled areas, and different markers
ax.plot(angles, values_1D_3D, color=colors[0], linewidth=2, marker=markers[6], markersize=7, label='1D + 3D (This work)')  # Star
ax.fill(angles, values_1D_3D, color=colors[0], alpha=0.1)
ax.plot(angles, values_3D, color=colors[3], linewidth=2, marker=markers[6], markersize=7, label='3D')  # Star
ax.fill(angles, values_3D, color=colors[3], alpha=0.1)
ax.plot(angles, values_3D_no_MF, color=colors[1], linewidth=2, marker=markers[5], markersize=7, label='3D no MF')
ax.fill(angles, values_3D_no_MF, color=colors[1], alpha=0.1)

ax.plot(angles, values_2D_no_MF, color=colors[4], linewidth=2, marker=markers[3], markersize=7, label='2D no MF')
ax.fill(angles, values_2D_no_MF, color=colors[4], alpha=0.1)
ax.plot(angles, values_2D, color=colors[2], linewidth=2, marker=markers[2], markersize=7, label='2D')
ax.fill(angles, values_2D, color=colors[2], alpha=0.1)
ax.plot(angles, values_1D_no_MF, color=colors[5], linewidth=2, marker=markers[1], markersize=7, label='1D no MF')
ax.fill(angles, values_1D_no_MF, color=colors[5], alpha=0.1)
ax.plot(angles, values_1D, color=colors[6], linewidth=2, marker=markers[0], markersize=7, label='1D')
ax.fill(angles, values_1D, color=colors[6], alpha=0.1)
# Adjust angle so "F1" is at the top
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Add additional concentric dashed circles at 75%, 50%, and 25% of the radius
for radius in [0.75, 0.5, 0.25]:
    ax.plot(np.linspace(0, 2 * np.pi, 100), [radius]*100, alpha=0.3, linestyle=(0, (5, 5)), color='grey', linewidth=1, zorder=0)

# Customize the outer circle with a dashed style and transparency
ax.spines['polar'].set_visible(True)
ax.spines['polar'].set_linestyle((0, (5, 5)))  # Dashed line for the outer circle
ax.spines['polar'].set_color('grey')
ax.spines['polar'].set_linewidth(1.5)
ax.spines['polar'].set_alpha(0.3)

# Hide y-axis labels and set the range to focus on the radar shape
ax.set_yticks([])
ax.set_ylim(0, 1)

# Set custom labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')  # Set font size for the axis labels
ax.tick_params(axis='x', labelsize=12)

# Add a custom legend without background and smaller text
ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), frameon=False, fontsize=12)

# Save the plot as a TIFF file with 300 DPI
plt.savefig('radar_1d.tiff', dpi=300, format='tiff')

# Show plot
plt.show()
