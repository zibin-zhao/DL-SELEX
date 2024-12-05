import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Define the labels for the radar chart
labels = ['F1', 'Recall', 'Specificity', 'Accuracy', 'NPV', 'PPV']
num_vars = len(labels)

# Updated values for each model
CHEMBERT_values = [0.7233, 0.7700, 0.5264, 0.8433, 0.8374, 0.8005]
DNABERT_values = [0.7426, 0.7900, 0.6347, 0.8100, 0.8768, 0.8105]
Co_attention_values = [0.8957, 0.8674, 0.8367, 0.8497, 0.8572, 0.8682]
Reverse_QKV_values = [0.8867, 0.8100, 0.8400, 0.8300, 0.8200, 0.8100]

# Create the angles for the radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # To close the circle

# Append the first value to the end to close the shape for each model
CHEMBERT_values += CHEMBERT_values[:1]
DNABERT_values += DNABERT_values[:1]
Co_attention_values += Co_attention_values[:1]
Reverse_QKV_values += Reverse_QKV_values[:1]

# Create the angles for the radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # To close the circle


# Set the font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = '#FFFFFF'  # Set background to white

# Define the color codes for each configuration
colors = ['#8E0F31',  # Dark Red
          '#024163',  # Dark Blue
          '#066190',  # Medium Blue
          '#77AECD']  # Light Blue # Gray, Dark Blue, Light Blue, Red

# Define the markers for each configuration
markers = ['^', 's', 'o', 'D']  # Triangle for Skip, Square for No-Skip

# Initialize the radar chart with a square aspect ratio
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Plot each configuration with transparency, filled areas, and different markers
ax.plot(angles, Co_attention_values, color=colors[0], linewidth=2, marker=markers[2], markersize=7, label='Co-attention (This work)')  # Square
ax.fill(angles, Co_attention_values, color=colors[0], alpha=0.1)

ax.plot(angles, CHEMBERT_values, color=colors[2], linewidth=2, marker=markers[0], markersize=7, label='CHEMBERT')  # Triangle
ax.fill(angles, CHEMBERT_values, color=colors[2], alpha=0.1)

ax.plot(angles, DNABERT_values, color=colors[1], linewidth=2, marker=markers[1], markersize=7, label='DNABERT')  # Square
ax.fill(angles, DNABERT_values, color=colors[1], alpha=0.1)

ax.plot(angles, Reverse_QKV_values, color=colors[3], linewidth=2, marker=markers[3], markersize=7, label='Reverse QKV')  # Square
ax.fill(angles, Reverse_QKV_values, color=colors[3], alpha=0.1)

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
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=12)
# Save the plot as a TIFF file with 300 DPI
plt.savefig('radar_chembert.tiff', dpi=300, format='tiff')

# Show plot
plt.show()
