#!/usr/bin/env python3
"""
Convert CSV file with sequences to FASTA format.
"""

import csv
import argparse


def csv_to_fasta(csv_file, fasta_file, seq_column='seq'):
    """
    Convert CSV file to FASTA format.
    
    Args:
        csv_file (str): Input CSV file path
        fasta_file (str): Output FASTA file path
        seq_column (str): Name of the column containing sequences (default: 'seq')
    """
    with open(csv_file, 'r') as csv_in, open(fasta_file, 'w') as fasta_out:
        reader = csv.DictReader(csv_in)
        
        for idx, row in enumerate(reader, start=1):
            sequence = row[seq_column]
            # Write FASTA header and sequence
            fasta_out.write(f'>seq_{idx}\n')
            fasta_out.write(f'{sequence}\n')
    
    print(f"Converted {idx} sequences from {csv_file} to {fasta_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert CSV file with sequences to FASTA format'
    )
    parser.add_argument(
        'csv_file',
        help='Input CSV file path'
    )
    parser.add_argument(
        'fasta_file',
        help='Output FASTA file path'
    )
    parser.add_argument(
        '--seq-column',
        default='seq',
        help='Name of the column containing sequences (default: seq)'
    )
    
    args = parser.parse_args()
    
    csv_to_fasta(args.csv_file, args.fasta_file, args.seq_column)

