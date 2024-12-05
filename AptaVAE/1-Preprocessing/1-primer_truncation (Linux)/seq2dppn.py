"""
    Convert sequence to dot parenthesis representation,
    Input: A text file containing a list of sequences with desired length
    Output: A CSV file containing the dot-parenthesis-plus notation of the sequences
    By Zibin Zhao
"""

from nupack import *
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_FILE = "./input/DHEAr7_SEQ.txt"
OUTPUT_FILE = "./output/DHEAr7_dppn.csv"

# Read the file and split it into sequences
with open(INPUT_FILE, "r") as file:
    data = file.read().split("\n")


#data = data[:25000]
my_model = Model(material='dna', celsius=25, sodium=0.147)

batch_size = 25000
total_batches = (len(data) + batch_size - 1) // batch_size  


print(f"Total number of sequences: {len(data)}")
print(f"Processing in {total_batches} batches...")

for batch in tqdm(range(total_batches), desc="Overall Progress"):
    start = batch * batch_size
    end = min((batch + 1) * batch_size, len(data))
    
    dppn_struct = []

    for i in tqdm(range(start, end), desc=f"Batch {batch+1}/{total_batches}"):
        mfe_structure = mfe(strands=data[i], model=my_model)
        dppn_struct.append(mfe_structure[0].structure.dotparensplus())

    # Convert to DataFrame and write to CSV
    df = pd.DataFrame(dppn_struct)
    if batch == 0:
        df.to_csv(OUTPUT_FILE, mode='w', header=None, index=False)
    else:
        df.to_csv(OUTPUT_FILE, mode='a', header=None, index=False)

    # Clear the list for the next batch
    dppn_struct.clear()

print(f"Processing partially complete. Please process to seq2dppn_part2.py")
