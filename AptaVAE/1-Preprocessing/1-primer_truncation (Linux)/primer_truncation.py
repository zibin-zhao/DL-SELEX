'''This script is designed for truncate the aptamer sequence to the minimal length
that can form the same secondary structure as described in the manuscript DL-SELEX by Zibin Zhao.'''

from nupack import *
import numpy as np
import pandas as pd

df_raw = pd.read_excel('ITC_truncation_2024.xlsx')
my_model = Model(material='dna', celsius=25, sodium=0.147)#, Potassium=0.0045)



print(df_raw.head())
def trim_sequence(seq, model):
    # Compute initial structure
    mfe_structures = mfe(strands=seq, model=model)
    initial_structure = mfe_structures[0].structure.dotparensplus()
    initial_count = (initial_structure.count('('), initial_structure.count(')'))

    # Trim from the start
    for i in range(len(seq)):
        trimmed_seq = seq[i:]
        mfe_structures = mfe(strands=trimmed_seq, model=model)
        structure = mfe_structures[0].structure.dotparensplus()
        count = (structure.count('('), structure.count(')'))
        print(f"Trimming from start: {trimmed_seq} -> {structure}")
        
        if count != initial_count:
            print(f"5'End before revert: {seq}")
            trimmed_seq = seq[i-1:]
            mfe_structures = mfe(strands=trimmed_seq, model=model)
            structure = mfe_structures[0].structure.dotparensplus()
            print(f"5'End after revert (i={i}): {trimmed_seq}")
            print(f"5'End after structure: {structure}")
            break

    # Trim from the end
    for j in range(1, len(trimmed_seq) + 1):
        trimmed_seq_end = trimmed_seq[:-j] 
        mfe_structures = mfe(strands=trimmed_seq_end, model=model)
        structure = mfe_structures[0].structure.dotparensplus()
        count = (structure.count('('), structure.count(')'))
        print(f"Trimming from end: {trimmed_seq_end} -> {structure}")
        
        if count != initial_count:
            print(f"3'End before revert: {trimmed_seq}")
            if j == 1:
                trimmed_seq_end = trimmed_seq
                print("No change in 3'end.")
                mfe_structures = mfe(strands=trimmed_seq_end, model=model)
                structure = mfe_structures[0].structure.dotparensplus()
                print(f"3'End after revert (j={j}): {trimmed_seq_end}")
                print(f"3'End after structure: {structure}")
                break
            else:
                trimmed_seq_end = trimmed_seq[:-(j-1)]
                mfe_structures = mfe(strands=trimmed_seq_end, model=model)
                structure = mfe_structures[0].structure.dotparensplus()
                print(f"3'End after revert (j={j}): {trimmed_seq_end}")
                print(f"3'End after structure: {structure}")
                break

    return trimmed_seq_end, structure

def process_sequences(df, model):
    modified_seqs = []
    dppns = []

    for index, row in df.iterrows():
        seq = row['Sequence']
        mod_seq, dppn = trim_sequence(seq, model)
        
        # Debugging print statements
        print(f"Processed sequence: {seq}")
        print(f"Modified sequence: {mod_seq}")
        print(f"Modified dppn: {dppn}")
        print("-" * 20)
        
        modified_seqs.append(mod_seq)
        dppns.append(dppn)

    df['modified seq'] = modified_seqs
    df['modified dppn'] = dppns

    return df

df = process_sequences(df, my_model)