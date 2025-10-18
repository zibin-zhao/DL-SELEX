# score_panel_aptadiff.py
import pandas as pd, numpy as np, joblib

EMB_PANEL = 'embed_CS_panel_RaptGen.csv'           # columns: (panel_id or index), seq, dim*
PANEL_IDS = 'CS_panel.csv'                          # panel_id,sequence,sequence_2d   (used only for IDs)
KDE_PATH  = 'CS_R3_RaptGen_latentKDE.joblib'
OUT_SCORES = 'scores_CS_panel_RaptGen.csv'

panel = pd.read_csv(EMB_PANEL)
ids_df = pd.read_csv(PANEL_IDS)                     # must contain 'panel_id' in panel order
ids = ids_df['panel_id'].values

bundle = joblib.load(KDE_PATH)
kde, dim_cols = bundle['kde'], bundle['dims']
Z = panel[dim_cols].to_numpy(float)

logdens = kde.score_samples(Z)
scores = -logdens                                    # higher = more binder-like

pd.DataFrame({'panel_id': ids, 'score': scores}).to_csv(OUT_SCORES, index=False)
print(f'Saved {OUT_SCORES} ({len(scores)} rows)')
