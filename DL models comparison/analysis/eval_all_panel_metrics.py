#eval_all_panel_metrics.py

import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

LABELS = 'CS_panel_labels.csv'
SCORE_FILES = {
    'AptaClux' : 'scores_CS_panel_clux.csv',
    'AptaDiff' : 'scores_CS_panel_aptadiff.csv',
    'RaptGen'  : 'scores_CS_panel_RaptGen.csv',
}
K_LIST = [1,3,5]

def eval_one(method, path):
    lab = pd.read_csv(LABELS)
    sc  = pd.read_csv(path)
    df  = lab.merge(sc, on='panel_id', how='inner').dropna()
    y = df['label'].astype(int).to_numpy()
    s = df['score'].astype(float).to_numpy()
    order = np.argsort(-s)
    auroc = roc_auc_score(y, s) if len(np.unique(y))==2 else float('nan')
    auprc = average_precision_score(y, s) if y.sum()>0 else float('nan')
    topk = {f'Top{k}': (y[order][:min(k,len(s))].sum()/min(k,len(s))) for k in K_LIST}
    return {'method':method, 'n':len(s), 'pos':int(y.sum()), 'neg':int((1-y).sum()),
            'AUROC':float(auroc), 'AUPRC':float(auprc), **topk}

rows = [eval_one(m, p) for m,p in SCORE_FILES.items()]
out = pd.DataFrame(rows)
out.to_csv('results_CS_panel_metrics.csv', index=False)
print(out)
