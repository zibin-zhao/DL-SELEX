import torch
from scipy.spatial.distance import cdist
from transformers import BertModel, BertTokenizer, BertConfig


# Load DNABERT model and tokenizer
config = BertConfig.from_pretrained("./dnabert/6mer/config.json")

# Load the DNABERT model
model = BertModel.from_pretrained("./dnabert/6mer/pytorch_model.bin", config=config)
tokenizer = BertTokenizer.from_pretrained("./dnabert/6mer/vocab.txt")  # Load the tokenizer used for encoding

# Load learned embeddings from the VAE decoder
embeddings = './data/seq_embeddings.pt'   #shape: (194, 5, 768)  # Load your learned embeddings

# Decoder output from VAE
decoder_output = ...  # Shape: (batch_size, sequence_length, latent_dim)

# Decode the sequences
decoded_sequences = []
for i in range(decoder_output.shape[0]):  # Iterate over each sequence in the batch
    sequence = []
    for j in range(decoder_output.shape[1]):  # Iterate over each vector in the sequence
        vector = decoder_output[i, j, :].unsqueeze(0)  # Shape: (1, latent_dim)
        
        # Find the closest embedding vector
        distances = cdist(vector.detach().numpy(), embeddings.detach().numpy(), metric='euclidean')
        closest_index = distances.argmin()
        
        # Map the embedding to the corresponding nucleotide
        nucleotide = tokenizer.convert_ids_to_tokens(closest_index)
        sequence.append(nucleotide)
    
    # Append the decoded sequence
    decoded_sequences.append(sequence)

# Convert the decoded sequences to string format
decoded_sequences = [''.join(seq) for seq in decoded_sequences]
