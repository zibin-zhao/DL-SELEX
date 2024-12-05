from transformers import BertModel, BertTokenizer, BertConfig
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch
import torch.nn.functional as F


# Load dataset
df = pd.read_csv('./data/our_dataset.csv')

smiles_target = df['Target_SMILES'].values

# Load the CHEMBERTa configuration
# Load the DNABERT model
model_name = 'seyonec/ChemBERTa-zinc-base-v1'

# Load pre-trained model and tokenizer
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.eval()

# Initialize an empty list to store the embeddings
smiles_embeddings = []

# Iterate over the SMILES strings in the dataframe
for smiles in df["Target_SMILES"]:
    print(smiles)
    # Tokenize the SMILES string
    tokens = tokenizer.encode(smiles, add_special_tokens=True)
    print(tokens)
    #print(tokens.shape)
    
    # Convert tokens to a tensor
    tensor = torch.tensor([tokens])  # Note: The input should be a list of lists
    tensor = pad_sequence(tensor, batch_first=True, padding_value=0)
    
    # Compute the embeddings
    with torch.no_grad():
        embedding = model(tensor)[0].squeeze(0)

    #embedding.append(embedding.squeeze(0))
    
    # Append the embedding to the list
    smiles_embeddings.append(embedding)

for i, embedding in enumerate(smiles_embeddings):
    print(f"Sequence {i}: {embedding.shape}")
    
max_chunks = max(tensor.shape[0] for tensor in smiles_embeddings)
#print(max_chunks)
padded_embeddings = []
for tensor in smiles_embeddings:
    #print("tensor shape: ", tensor.shape)
    padding = torch.zeros(max_chunks - tensor.shape[0], 768)
    #print("padding shape: ", padding.shape)
    padded_tensor = torch.cat((tensor, padding), dim=0)
    padded_embeddings.append(padded_tensor)
padded_target_embeddings = torch.stack(padded_embeddings)

#print(padded_target_embeddings.shape)

# flat_embeddings = padded_embeddings.reshape(len(embeddings), -1)
# mean_embeddings = torch.stack([tensor.mean(dim=0) for tensor in embeddings])

smiles_embeddings = padded_target_embeddings


# Assume that `smiles_embeddings` is a tensor of shape (194, 45, 768)

# Permute the tensor to move the sequence dimension to the end
smiles_embeddings = smiles_embeddings.permute(0, 2, 1)  # shape: (194, 768, 45)

# Apply average pooling
smiles_embeddings = F.avg_pool1d(smiles_embeddings, kernel_size=max_chunks//1)  # shape: (194, 768, 5)

# Permute the tensor to move the sequence dimension back to the middle
smiles_embeddings = smiles_embeddings.permute(0, 2, 1)

print(smiles_embeddings.shape)

# Save the tensor to a file
torch.save(smiles_embeddings, './data/tar_embeddings.pt')