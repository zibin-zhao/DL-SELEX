import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Data preparation
data = {
    'Model': [
        'This Work', 'Random Forest', 'LightGBM', 'Naive Bayes', 
        'Ada Boost', 'Logistic Regression', 'LDA', 'SVM'
    ],
    'Accuracy': [0.8497, 0.6577, 0.6346, 0.6027, 0.5115, 0.5571, 0.5423, 0.4258],
    'AUC': [0.9834, 0.6924, 0.6782, 0.6822, 0.5715, 0.5378, 0.6041, 0.4357],
    'Recall': [0.8674, 0.6574, 0.6346, 0.6027, 0.5115, 0.5571, 0.5423, 0.4258],
    'Precision': [0.8862, 0.6515, 0.5975, 0.6854, 0.5008, 0.5146, 0.5160, 0.3856],
    'F1': [0.9611, 0.6351, 0.5978, 0.6094, 0.4728, 0.5101, 0.4917, 0.3663]
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# Custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#c42238'])

# Plotting
plt.figure(figsize=(7, 3))  # Square figure size
sns.set(font_scale=0.6)  # Reduce font size further
sns.set_style("white")  # Clean style
plt.rcParams["font.family"] = "Arial"  # Set font to Arial

# Heatmap
ax = sns.heatmap(
    df.T,  # Transpose to make it horizontal
    annot=True,  # Remove numbering inside cells
    cmap=custom_cmap,  # Custom colormap
    linewidths=0.4,  # Black outline thickness
    linecolor='white',  # Outline color
    cbar_kws={'label': 'Metric Value'},  # Colorbar label
    square=True  # Force square cells
)

# Adjust aspect ratio
ax.set_aspect('auto')  # Ensure squares remain consistent

# Titles and labels
ax.xaxis.set_label_position('top')  # Move x-axis title to the top
ax.xaxis.tick_top() 
ax.tick_params(axis='x', length=0)  # Remove tick marks on the x-axis

plt.xticks(rotation=90, fontsize=8,  fontweight='bold')  # Adjust label rotation and font size
plt.yticks(fontsize=8, rotation=0, fontweight='bold')
# Place x-axis title at the top
 # Move x-axis ticks to the top
#plt.xlabel('Model', fontsize=10, fontweight='bold')  # Set x-axis title


plt.tight_layout()  # Adjust layout to fit everything
plt.savefig('model_comp_heatmap.tiff', dpi=300, format='tiff', facecolor='#FFFFFF')
plt.show()
