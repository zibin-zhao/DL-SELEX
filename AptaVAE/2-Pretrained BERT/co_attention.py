'''Co-attention methcanism using DNABERT and CHEMBERT 
    embeddings for the steroid targets by Zibin Zhao '''

import torch
import torch.nn as nn

class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_transform = nn.Linear(hidden_dim, hidden_dim)
        self.k_transform = nn.Linear(hidden_dim, hidden_dim)
        self.v_transform = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v):
        # Compute the attention weights
        attn_weights = self.softmax(torch.bmm(self.q_transform(q), self.k_transform(k).transpose(1, 2)))
        
        # Apply the attention weights to v
        attn_output = torch.bmm(attn_weights, self.v_transform(v))
        
        # Add the original query to the attention output (residual connection)
        output = q + attn_output
        
        return output, attn_weights

    
    
# class MultipleCoAttention(nn.Module):
#     def __init__(self, d_model, num_heads, num_parts):
#         super().__init__()
#         self.num_parts = num_parts
#         self.coattn_layers = nn.ModuleList([
#             CoAttention(d_model, num_heads) for _ in range(num_parts)
#         ])

#     def forward(self, q, k):
#         # Divide the longer sequence into parts
#         k_parts = torch.chunk(k, self.num_parts, dim=1)

#         outputs = []
#         for i in range(self.num_parts):
#             # Apply co-attention to each part
#             output, _ = self.coattn_layers[i](q, k_parts[i], k_parts[i])
#             outputs.append(output)

#         # Concatenate the outputs along the sequence dimension
#         output = torch.cat(outputs, dim=1)

#         return output


# load tensor
seq_embeddings = torch.load('./data/seq_embeddings.pt')  # Shape: (batch_size, dna_seq_len, hidden_dim) (194, 5, 768)
tar_embeddings = torch.load('./data/tar_embeddings.pt')  # Shape: (batch_size, smiles_seq_len, hidden_dim) (194, 5, 768)


# Initialize the CoAttention module
co_attn = CoAttention(hidden_dim=768)

# Apply the co-attention mechanism query = target, key = seq, value = seq, according to target, what sequence should we have 
output, attn_weights = co_attn(tar_embeddings, seq_embeddings, seq_embeddings)   
print("output shape:", output.shape)
print("attenion weights shape: ", attn_weights.shape)

torch.save(output, './model/co_attn_output.pt')
torch.save(attn_weights, './model/co_attn_weights.pt')