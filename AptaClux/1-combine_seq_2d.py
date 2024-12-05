"""
    This script is used to combine the sequence file and the secondary structure file into a single file.
    input: sequence file .txt & dppn file.csv
    output: combined file .csv"""

import pandas as pd


INPUT_SEQ = './raw_NGS/DHEA/r3/DHEAr3_SEQ.txt'
INPUT_DPPN = './raw_NGS/DHEA/r3/DHEAr3_dppn.csv'
OUTPUT_NAME = './raw_NGS/DHEA/r3/DHEAr3_combined.csv'

# Load the TXT file containing sequences
seq_df = pd.read_csv(INPUT_SEQ, header=None, names=['Sequence'])
print(f"length of sequence: {len(seq_df)}")
# Load the CSV file containing secondary structures
# Address the DtypeWarning by setting low_memory=False
seq2d_df = pd.read_csv(INPUT_DPPN, header=None, names=['Sequence_2d'], low_memory=False)
print(f"length of 2d: {len(seq2d_df)}")
# Reset the index of both DataFrames to ensure they are aligned
seq_df.reset_index(drop=True, inplace=True)
seq2d_df.reset_index(drop=True, inplace=True)

# Check if both DataFrames have the same number of rows
if len(seq_df) == len(seq2d_df):
    # Concatenate the DataFrames
    combined_df = pd.concat([seq_df, seq2d_df], axis=1)
else:
    print("Error: The number of rows in the sequence file and the secondary structure file do not match.")

# Save the combined DataFrame
combined_df.to_csv(OUTPUT_NAME, index=False)

