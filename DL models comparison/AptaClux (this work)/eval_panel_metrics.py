import pandas as pd, numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score 

LABELS = 'CS_panel_labels.csv' # panel_id,label (2 pos, 6 neg) 
SCORES = 'scores_CS_panel_clux.csv' # panel_id,score 
K = [1,3,5] 

lab = pd.read_csv(LABELS) 
sc = pd.read_csv(SCORES) 
df = lab.merge(sc, on='panel_id', how='inner').dropna() 

y = df['label'].astype(int).to_numpy() 
s = df['score'].astype(float).to_numpy() 

auroc = roc_auc_score(y, s) if len(np.unique(y))==2 else float('nan') 
auprc = average_precision_score(y, s) if y.sum()>0 else float('nan') 
order = np.argsort(-s) 
topk = lambda k: y[order][:min(k,len(s))].sum()/min(k,len(s)) 

print({'n':len(s), 'pos':int(y.sum()), 'neg':int((1-y).sum()), 'AUROC': auroc, 'AUPRC': auprc, **{f'Top{k}': topk(k) for k in K}}) 