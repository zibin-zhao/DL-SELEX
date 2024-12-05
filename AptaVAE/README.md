
# AptaVAE Workflow

This repository contains scripts and files for the **AptaVAE** model in the **DL-SELEX** project. The workflow involves sequence processing and feature extraction, followed by data preparation for machine learning models.

## Overview

The workflow is divided into the following steps:
1. **Primer Truncation**
2. **1D and 3D Preprocessing**
3. **Model Training and Evaluation**
4. **BGA Analysis and SELEX Library Design**

## Quickstart
Researchers who wish to apply the cerrent workflow to other types of molecule family, simply replace the colelcted aptamer sequences with their correponding classes and scores in the train_dataset.xlsx under the 0-Sample Data folder. Then, the predicted initial library can be obtained following the workflow in the followings.

## Workflow

Notes: all scripts were tested and can be run across windows, linux and macOS platform, except for the primer_truncation script that utilzied the NUPACK packages. Before running primer_truncation.py, users should make sure the NUPACK can be run locally suggested on Linux or macOS. Detailed NUPACK installation can be found at: https://docs.nupack.org/ with proper license.

### 1. Primer Truncation

Script: `Primer_truncation.py`

- **Input:** Sequence data (.xlsx)
- **Output:** Truncated sequences and corresponding DPPN
- **Details:** Global truncation program designed for cutting the redundant primer to improve model accuracy.

### 2. MF_preprocessing (1D)

Script: `MF_preprocessing.py`

- **Input:** `train_dataset_refined_primer.csv`
- **Outputs:** 
  - `MF_primer_input.pt`
  - `MF_primer_mask.pt`
- **Details:** Processes raw data to extract 1D features for machine learning. The following data columns are used:
  - Target
  - Class
  - SMILES
  - Modified Sequence
  - Relative Score
  - Modified DPPN

### 3. 3D_preprocessing (3D)

Script: `3D_preprocessing.py`

- **Input:** `train_dataset_refined_primer.csv`
- **Outputs:** 
  - `3D_primer_input.pt`
  - `3D_primer_mask.pt`
- **Details:** Processes raw data to extract 3D structural features for machine learning.

### 4. Preprocessing and Model Training

Script: `trained_refined_primer.py`

- **Inputs:** 
  - `MF1D_input.pt`, `MF1D_mask.pt`
  - `3D_primer_input.pt`, `3D_primer_mask.pt`
- **Outputs:** 
  - Saved model
  - Training logs
  - Test loss, edit distance, accuracy, AUROC, and F1 (1-vs-all) metrics

### 5. BGA Analysis

Script: `BGA_analysis.py`

- **Input:** Model latent space
- **Output:** Sequences with corresponding class and scores

### 6. Decoded Sequences by Class

Script: `decoded_seq_by_class.py`

- **Input:** BGA Analysis output
- **Output:** Ranked class sequences

### 7. MSA for Class Sequences

Script: `MSA_class_sequences.py`

- **Details:** Conducts multiple sequence alignment (MSA) for sequences in each class, preparing for SELEX library design.

### 8. SELEX Library Design

Notebook: `SELEX_library_design.ipynb`

- **Input:** CSV file from MSA results (Mafft)
- **Output:** Initial SELEX library design

## Requirements

- Python 3.8 or higher
- PyTorch
- Biopython
- Nupack (Linux)

## License

This project is licensed under the MIT License.

## Contact

For further questions, please contact zzhaobz@connect.ust.hk.
