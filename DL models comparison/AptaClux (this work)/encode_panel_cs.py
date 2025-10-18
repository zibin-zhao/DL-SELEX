import pandas as pd 
import numpy as np 
import torch 

MAX_LENGTH = 39 # CS 
MAP_SEQ = {'A':0,'C':1,'G':2,'T':3} 
MAP_DB = {'.':0,'(':1,')':2} 

def one_hot_seq(seq): 
    s = (seq.strip().upper() + 'A'*MAX_LENGTH)[:MAX_LENGTH] 
    X = np.zeros((MAX_LENGTH, 4), dtype=np.float32) 
    for i,ch in enumerate(s): 
        X[i, MAP_SEQ.get(ch, 0)] = 1.0 # unknown -> 'A' 
    return X.reshape(-1) # 4L 

def one_hot_db(db): 
    d = (db.strip() + '.'*MAX_LENGTH)[:MAX_LENGTH] 
    X = np.zeros((MAX_LENGTH, 3), dtype=np.float32) 
    for i,ch in enumerate(d): 
        X[i, MAP_DB.get(ch, 0)] = 1.0 # unknown -> '.' 
    return X.reshape(-1) # 3L 

def main(in_csv='CS_panel.csv', out_pt='data/CS_panel_input.pt'): 
    df = pd.read_csv(in_csv) # expects: panel_id,sequence,sequence_2d 
    X = np.stack([ 
        np.concatenate([one_hot_seq(s), one_hot_db(d)]) 
        for s, d in zip(df['Sequence'], df['Sequence_2d']) 
    ]) 
    X = torch.from_numpy(X) # shape (8, 273) 
    assert X.shape[1] == 273, f'Expected 273 features, got {X.shape[1]}' 
    torch.save(X, out_pt) 
    print(f'Saved {out_pt} with shape {tuple(X.shape)}') 

if __name__ == '__main__': 
    main() 