import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('./8RP_BGA_decoded_results01.csv')
FILE = '8RP_BGA_grouped_sequences_by_classes01.xlsx' #*path to save your output

start_str = '<'
end_str = '>'

def extract_sequences(seq):
    valid_sequences = []
    recording = False
    current_sequence = ''
    
    for char in seq:
        if char == '<':
            if recording:
                current_sequence = ''
            else:
                recording = True
        elif char == '>':
            if recording:
                if len(current_sequence) >= 10:
                    valid_sequences.append(current_sequence)
                recording = False
                current_sequence = ''
        elif recording:
            current_sequence += char
    
    return valid_sequences

df['decoded_seqs'] = df['Decoded Sequences'].apply(extract_sequences)
df = df.explode('decoded_seqs')
df = df[df['decoded_seqs'].str.len() > 0]

def sort_sequences_by_score(group_df):
    """
    Sort sequences based on their score within a group in descending order.
    """
    return group_df.sort_values(by='Decoded Scores', ascending=False)

# Group the DataFrame by the 'Decoded Classes' column and apply the sort function
df_sorted = df.groupby('Decoded Classes').apply(sort_sequences_by_score).reset_index(drop=True)

# Save the sorted data to Excel
with pd.ExcelWriter(FILE) as writer:
    for name, group in df_sorted.groupby('Decoded Classes'):
        group.to_excel(writer, sheet_name=name, index=False)
