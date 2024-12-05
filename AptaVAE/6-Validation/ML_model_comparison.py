'''using pycaret for comparing the same dataset 
    to other machine learning models by Zibin Zhao'''

import matplotlib.pyplot as plt
import numpy as np

# Set a professional and minimalistic style
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Data setup
models = ["Our's", 'RF', 'GB', 'DT', 'LGBM', 'ET', 'NB', 'KNN', 'LR', 'LDA', 'Ridge', 'Ada', 'SVM', 'Dummy', 'QDA']
accuracy = [0.9737,0.6577, 0.6423, 0.6346, 0.6346, 0.6192, 0.6027, 0.5654, 0.5571, 0.5423, 0.5269, 0.5115, 0.4258, 0.3203, 0.0500]
auc = [0.98,0.0646, 0.0671, 0.0605, 0.0716, 0.0654, 0.0658, 0.0590, 0.0579, 0.0594, 0.0000, 0.0534, 0.0000, 0.0500, 0.0713]
f1 = [0.9611,0.6351, 0.6153, 0.6052, 0.5978, 0.5921, 0.6094, 0.5527, 0.5101, 0.4917, 0.4438, 0.4728, 0.3663, 0.1560, 0.0452]
x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

# Colors inspired by Prism's default palettes
colors = ['#1f77b4', '#aec7e8', '#ffbb78']

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, accuracy, width, label='Accuracy', color=colors[0])
rects2 = ax.bar(x, auc, width, label='AUC', color=colors[1])
rects3 = ax.bar(x + width, f1, width, label='F1 Score', color=colors[2])

# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.legend(frameon=True, loc='best')

# Display the values on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
