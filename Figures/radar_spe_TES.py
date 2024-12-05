import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data setup for TES-R7MP-T and AI-TESR3R5HD-T
categories = ['TES', 'CS', 'BE', 'DHEA', 'CHO', 'PRO']
labels = ['TES-R7MP-T', 'AI-TESR3R5HD-T']

# Normalized values for each sequence
data = {
    'TES-R7MP-T': [1, 0.0045, 0.01, 0.0024,	0.0066, 0.0108
],
    'AI-TES-R3R5HD-T': [1,0.0146,0.0519, 0.0015,	0.0031,	0.0344]
}

# Number of variables
num_vars = len(categories)

# Angles for radar chart
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

# Set the font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = '#FFFFFF'  # Set background to white

# Number of variables
num_vars = len(categories)

# Angles for radar chart
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

# Define colors and markers for different sequences
colors = ['#8E0F31',  # Dark Red
          '#024163',  # Dark Blue
          '#066190',  # Medium Blue
          '#77AECD']  # Light Blue # Gray, Dark Blue, Light Blue, Red

markers = ['^', 's', 'D', 'o']  # Circle, Square, Diamond, Triangle

# Create the radar chart with a square aspect ratio
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# Plot each model with different symbols, transparency, and filled areas
for idx, (label, values) in enumerate(data.items()):
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, color=colors[idx], linewidth=2, marker=markers[idx], markersize=7, label=label)
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# Adjust angle so "CS" is at the top
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Set custom labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

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
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')  # Set font size for the axis labels
ax.tick_params(axis='x', labelsize=12)

# Add a custom legend without background and smaller text
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=12)

# Save the plot as a TIFF file with 300 DPI
plt.savefig('radar_spe_TES.tiff', dpi=300, format='tiff', facecolor='#FFFFFF')

# Show plot
plt.show()
