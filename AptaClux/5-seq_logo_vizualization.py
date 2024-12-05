'''This script is designed to draw sequence logo map for the 
clustered sequences for visualization purpose in SI, by Zibin Zhao'''

from Bio import SeqIO
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt

def parse_fasta(fasta_file):
    """Parse sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def calculate_frequency_matrix(sequences):
    """Calculate the frequency matrix from a list of sequences."""
    # Get the length of the sequences
    sequence_length = len(sequences[0])

    # Initialize a frequency matrix with zeros
    bases = ['A', 'C', 'G', 'T']
    freq_matrix = pd.DataFrame(0, index=bases, columns=range(sequence_length))

    # Count occurrences of each base at each position
    for seq in sequences:
        for i, base in enumerate(seq):
            if base in bases:
                freq_matrix.at[base, i] += 1

    # Normalize to get frequencies
    freq_matrix = freq_matrix.div(freq_matrix.sum(axis=0), axis=1)

    return freq_matrix

def create_sequence_logo(freq_matrix, output_file):
    """Create and save a sequence logo."""
    # Create the logo
    logo = lm.Logo(freq_matrix.T)

    # Customize the appearance
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(rotation=90, fmt="%d", anchor=0)

    # Save the logo to a file
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Sequence logo saved to {output_file}")

def main():
    # Specify input and output paths here
    NAME = 'cc_CSr3'
    input_file = f"{NAME}.fasta"
    output_file = f"{NAME}.png"

    # Parse sequences from the input FASTA file
    sequences = parse_fasta(input_file)

    # Check if sequences are of the same length
    if len(set(len(seq) for seq in sequences)) > 1:
        raise ValueError("All sequences must have the same length.")

    # Calculate the frequency matrix
    freq_matrix = calculate_frequency_matrix(sequences)

    # Create and save the sequence logo
    create_sequence_logo(freq_matrix, output_file)

if __name__ == "__main__":
    main()