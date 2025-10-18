# fit_kde_aptadiff.py
import pandas as pd, numpy as np, joblib
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold

SEED = 42
EMB_R3 = 'embed_CS_R3_RaptGen.csv'                 # has columns: index,seq,dim*
OUT_KDE = 'CS_R3_RaptGen_latentKDE.joblib'

r3 = pd.read_csv(EMB_R3)
dim_cols = [c for c in r3.columns if c.startswith('dim')]
Z = r3[dim_cols].to_numpy(float)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(0.05, 1.5, 30)},
                    cv=cv, n_jobs=-1)
grid.fit(Z)
joblib.dump({'kde': grid.best_estimator_, 'bw': grid.best_params_['bandwidth'], 'dims': dim_cols}, OUT_KDE)
print(f'Saved {OUT_KDE}; best bw={grid.best_params_["bandwidth"]}; dims={len(dim_cols)}')
