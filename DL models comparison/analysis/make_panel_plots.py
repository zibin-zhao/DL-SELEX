# make_panel_plots.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

LABELS_CSV = 'CS_panel_labels.csv'         # panel_id,label
SCORES_GLOB = 'scores_CS_panel_*.csv'      # e.g., *_clux.csv, *_aptadiff.csv, *_raptgen.csv
OUT_DIR = 'figures'
K_LIST = [1,3,5]

# Color palette for better contrast
COLORS = {
    'clux': '#024163',      # Dark blue
    'aptadiff': '#C42238',  # Red
    'RaptGen': '#066190',   # Teal blue
    'default1': '#8E0F31',  # Dark red
    'default2': '#D98380',  # Light pink
    'default3': '#77AECD',  # Light blue
}

# Colors for positive/negative labels
COLOR_NEGATIVE = '#95A3A6'  # Gray
COLOR_POSITIVE = '#E74C3C'  # Bright red

os.makedirs(OUT_DIR, exist_ok=True)
lab = pd.read_csv(LABELS_CSV)
assert {'panel_id','label'}.issubset(lab.columns)

# load scores from multiple methods
score_files = sorted(glob.glob(SCORES_GLOB))
methods = []
all_metrics = []

def get_method_color(method_name, index=0):
    """Get color for a method, with fallback to defaults."""
    method_lower = method_name.lower()
    if method_lower in COLORS:
        return COLORS[method_lower]
    for key in COLORS:
        if key in method_lower or method_lower in key:
            return COLORS[key]
    # Fallback to default colors
    defaults = ['default1', 'default2', 'default3']
    return COLORS[defaults[index % len(defaults)]]

def compute_metrics(df):
    y = df['label'].to_numpy(dtype=int)
    s = df['score'].to_numpy(dtype=float)
    res = {'n': len(y), 'pos': int(y.sum()), 'neg': int((1-y).sum())}
    res['AUROC'] = roc_auc_score(y, s) if len(np.unique(y))==2 else np.nan
    res['AUPRC'] = average_precision_score(y, s) if y.sum()>0 else np.nan
    order = np.argsort(-s)
    for k in K_LIST:
        k_eff = min(k, len(y))
        res[f'Top{k}'] = y[order][:k_eff].sum()/k_eff
    return res

# Collect ranking frames for plotting
rank_frames = []

for path in score_files:
    meth = os.path.basename(path).replace('scores_CS_panel_','').replace('.csv','')
    df = pd.read_csv(path).merge(lab, on='panel_id', how='inner')
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df)+1)
    df['method'] = meth
    rank_frames.append(df[['method','rank','panel_id','score','label']])

    m = compute_metrics(df)
    m['method'] = meth
    methods.append(meth)
    all_metrics.append(m)

rank_df = pd.concat(rank_frames, ignore_index=True)
metrics_df = pd.DataFrame(all_metrics)

# ---- Figure 1: Ranking strip plot ----
# One row per method; x = rank (1..8); y = score (scaled), color by label
fig, axes = plt.subplots(len(methods), 1, figsize=(8, 2*len(methods)), sharex=True, facecolor='white')
if len(methods)==1:
    axes = [axes]
for ax, meth in zip(axes, methods):
    d = rank_df[rank_df['method']==meth]
    # plot negatives
    d0 = d[d['label']==0]
    ax.scatter(d0['rank'], d0['score'], s=80, c=COLOR_NEGATIVE, 
               label='Negative', alpha=0.85, edgecolors='white', linewidth=1.5)
    # plot positives
    d1 = d[d['label']==1]
    ax.scatter(d1['rank'], d1['score'], s=120, c=COLOR_POSITIVE, 
               marker='D', label='Positive', alpha=0.95, edgecolors='white', linewidth=1.5)
    
    # Add value annotations for all points
    for _, row in d.iterrows():
        # Offset annotation position based on label
        y_offset = 0.05 if row['label'] == 1 else -0.05
        ax.annotate(f"{row['score']:.3f}", 
                   xy=(row['rank'], row['score']), 
                   xytext=(0, y_offset), 
                   textcoords='offset points',
                   ha='center', va='bottom' if row['label'] == 1 else 'top',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    ax.set_ylabel(meth, rotation=0, ha='right', va='center', fontsize=11, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.3, color='gray')
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[-1].set_xlabel('Rank (higher score → left)', fontsize=11, fontweight='bold')
axes[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'CS_panel_ranking.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Figure 2: Metrics bar chart ----
metrics_long = []
for _, row in metrics_df.iterrows():
    for m in ['AUROC','AUPRC'] + [f'Top{k}' for k in K_LIST]:
        metrics_long.append({'method': row['method'], 'metric': m, 'value': row[m]})
metrics_long = pd.DataFrame(metrics_long)

# order metrics nicely
metric_order = ['AUROC','AUPRC'] + [f'Top{k}' for k in K_LIST]
x_pos = np.arange(len(metric_order))
width = 0.8 / max(1, len(methods))

fig, ax = plt.subplots(figsize=(9, 4 + 0.6*len(methods)), facecolor='white')
for i, meth in enumerate(methods):
    vals = [metrics_long[(metrics_long.method==meth) & (metrics_long.metric==m)]['value'].values[0]
            for m in metric_order]
    color = get_method_color(meth, i)
    bars = ax.bar(x_pos + i*width, vals, width, label=meth, color=color, 
                  alpha=0.9, edgecolor='white', linewidth=1.5)
    
    # Add value annotations on top of each bar
    for j, (bar, val) in enumerate(zip(bars, vals)):
        if not np.isnan(val):  # Only annotate if value is not NaN
            ax.annotate(f'{val:.3f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                       xytext=(0, 3), 
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, 
                                edgecolor=color, linewidth=1))

ax.set_xticks(x_pos + (len(methods)-1)*width/2)
ax.set_xticklabels(metric_order, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.15)  # Increased to accommodate annotations
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=True, shadow=True, ncol=min(len(methods),3), fontsize=10)
ax.set_title('CS Panel Metrics (n=8; 2 pos, 6 neg)', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle=':', alpha=0.3, color='gray')
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'CS_panel_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Optional Figure 3: ROC (coarse with n=8) ----
fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor='white')
ax.plot([0,1],[0,1], linestyle='--', linewidth=2, color='gray', alpha=0.7, label='Random')

for i, meth in enumerate(methods):
    d = rank_df[rank_df['method']==meth]
    fpr, tpr, _ = roc_curve(d['label'], d['score'])
    auroc = metrics_df[metrics_df.method==meth]['AUROC'].values[0]
    color = get_method_color(meth, i)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=3, color=color, alpha=0.9, 
            label=f"{meth} (AUROC = {auroc:.3f})")
    
    # Add data points with annotations
    for j, (fp, tp) in enumerate(zip(fpr, tpr)):
        if j < len(fpr) - 1:  # Don't annotate the last point (1,1)
            ax.scatter(fp, tp, s=60, color=color, alpha=0.8, edgecolors='white', linewidth=1.5)
            # Add threshold value annotation
            threshold = d['score'].iloc[j] if j < len(d) else d['score'].iloc[-1]
            ax.annotate(f'{threshold:.3f}', 
                       xy=(fp, tp), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, 
                                edgecolor=color, linewidth=1))

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - CS Panel (n=8; 2 pos, 6 neg)', fontsize=13, fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle=':', color='gray')
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'CS_panel_roc.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Figure 4: PRC (Precision-Recall) Curve ----
fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor='white')

# Calculate baseline (random classifier performance)
baseline_precision = metrics_df['pos'].iloc[0] / (metrics_df['pos'].iloc[0] + metrics_df['neg'].iloc[0])
ax.axhline(y=baseline_precision, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Random (AP = {baseline_precision:.3f})')

for i, meth in enumerate(methods):
    d = rank_df[rank_df['method']==meth]
    precision, recall, _ = precision_recall_curve(d['label'], d['score'])
    auprc = metrics_df[metrics_df.method==meth]['AUPRC'].values[0]
    color = get_method_color(meth, i)
    
    # Plot PRC curve
    ax.plot(recall, precision, linewidth=3, color=color, alpha=0.9, 
            label=f"{meth} (AP = {auprc:.3f})")
    
    # Add data points with annotations
    for j, (rec, prec) in enumerate(zip(recall, precision)):
        if j < len(recall) - 1:  # Don't annotate the last point
            ax.scatter(rec, prec, s=60, color=color, alpha=0.8, edgecolors='white', linewidth=1.5)
            # Add threshold value annotation
            threshold = d['score'].iloc[j] if j < len(d) else d['score'].iloc[-1]
            ax.annotate(f'{threshold:.3f}', 
                       xy=(rec, prec), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, 
                                edgecolor=color, linewidth=1))

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - CS Panel (n=8; 2 pos, 6 neg)', fontsize=13, fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(alpha=0.3, linestyle=':', color='gray')
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'CS_panel_prc.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- Save numeric table (SI) ----
# si = metrics_df[['method','AUROC','AUPRC','Top1','Top3','Top5','pos','neg']]
# si.to_csv('si/Table_SX_CS_panel_metrics.csv', index=False)
# print('Wrote figures and SI table.')
