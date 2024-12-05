import pandas as pd
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F

# Load your dataset
df = pd.read_csv('./data/our_dataset.csv')

# Display the first few rows of the dataframe
# print(df.head())


def chunk_sequence(sequence, k):
    return [sequence[i:i+k] for i in range(0, len(sequence), k)]

# Apply the chunking function to each sequence
df['6mers'] = df['Sequence'].apply(lambda x: chunk_sequence(x, 6))

# Display the first few rows of the dataframe
# print(df.head())


# Load the DNABERT configuration
#config = BertConfig.from_pretrained("./dnabert/6mer/config.json")

# Load the DNABERT model
model_name = 'zhihan1996/DNA_bert_6'

# Load pre-trained model and tokenizer
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.eval()


# Tokenize the chunks and compute the DNABERT embeddings
embeddings = []
for chunks in df['6mers']:
    # Tokenize the chunks
    tokens = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks]
    print(tokens)
    
    # Convert tokens to tensors and pad them
    tensors = [torch.tensor(token) for token in tokens]
    tensor = pad_sequence(tensors, batch_first=True, padding_value=0)
    
    # Compute the DNABERT embeddings
    with torch.no_grad():
        embedding = model(tensor)[0]
    
    embeddings.append(embedding.squeeze(1))


# Verify the shapes
for i, embedding in enumerate(embeddings):
    print(f"Sequence {i}: {embedding.shape}")
    #print(embedding)

max_chunks = max(tensor.shape[0] for tensor in embeddings)
padded_embeddings = []
for tensor in embeddings:
    #print("tensor shape: ", tensor.shape)
    padding = torch.zeros(max_chunks - tensor.shape[0], 768)
    #print("padding shape: ", padding.shape)
    padded_tensor = torch.cat((tensor, padding), dim=0)
    padded_embeddings.append(padded_tensor)
padded_embeddings = torch.stack(padded_embeddings)



# Assume that `smiles_embeddings` is a tensor of shape (194, 45, 768)

# Permute the tensor to move the sequence dimension to the end
seq_embeddings = padded_embeddings.permute(0, 2, 1)  # shape: (194, 768, 45)

# Apply average pooling
seq_embeddings = F.avg_pool1d(seq_embeddings, kernel_size=max_chunks//1)  # shape: (194, 768, 5)

# Permute the tensor to move the sequence dimension back to the middle
seq_embeddings = seq_embeddings.permute(0, 2, 1)  # shape: (194, 20, 768)
#seq_embeddings = seq_embeddings.squeeze(2)

print(seq_embeddings.shape)

# save tensor
torch.save(seq_embeddings, './data/seq_embeddings.pt')
