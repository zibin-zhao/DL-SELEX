import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import nn, optim
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Read the sequences and embeddings
sequences = pd.read_csv('./data/our_dataset.csv')['Sequence'].tolist()
embeddings = torch.load('./data/seq_embeddings.pt').reshape(-1, 768) # Flatten the embeddings

# One-hot encode the sequences. Assume all sequences have same length
onehot_encoder = OneHotEncoder(sparse=True)
onehot_encoded = onehot_encoder.fit_transform(np.array(sequences).reshape(-1, 1))

class Seq2EmbDataset(Dataset):
    def __init__(self, sequences, embeddings):
        self.sequences = sequences
        self.embeddings = embeddings

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.embeddings[idx]

dataset = Seq2EmbDataset(onehot_encoded, embeddings)

# Define the model
class SeqGenerationModel(nn.Module):
    def __init__(self, emb_dim, seq_len, vocab_size):
        super(SeqGenerationModel, self).__init__()
        self.linear = nn.Linear(emb_dim, seq_len*vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return x.view(x.shape[0], seq_len, vocab_size)

# Define the training loop
def train(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for sequences, embeddings in dataloader:
            optimizer.zero_grad()
            sequences_hat = model(embeddings)
            loss = criterion(sequences_hat, torch.argmax(sequences, dim=1))
            loss.backward()
            optimizer.step()

# Initialize everything and start training
seq_len = 45
vocab_size = 1
model = SeqGenerationModel(768, seq_len, vocab_size) # define seq_len and vocab_size according to your data
dataloader = DataLoader(dataset, batch_size=32)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train(model, dataloader, optimizer, criterion, epochs=10)
