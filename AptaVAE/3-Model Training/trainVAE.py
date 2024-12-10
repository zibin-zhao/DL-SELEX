'''Main training pipeline for the VAE model by Zibin Zhao'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
# Local application imports
from utility import *


# Hyperparameter tuning
LAYER_1_SIZE = 256
LAYER_2_SIZE = 256
DROPOUT_RATE = 0.75
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
TRAIN_SPLIT = 0.6

EPOCHS = 8000
WEIGHT_DECAY = 1e-4 

# Paths
WRITER_PATH = "runs/trainVAE/test_model" 
MODEL_NAME = './save_models/test_model.pt'

# Model parameters
INPUT_SIZE = 1741
OUTPUT_SIZE = 1741   # maximum length of sequence + SOS + EOS + class + score
LOG_INTERVAL = 1000
EARLY_LIMIT = 500
#N_NEW_SEQ = 10
SEQ_LENGTH = 118 * 6 #826
MAX_LENGTH = 118    # maximum length of sequence + SOS + EOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
SEED = 42


# Initialize TensorBoard writer
writer = SummaryWriter(WRITER_PATH)

#* Please change path according to your case
DATA_PATH = './data/MF_input_data.pt'
MASK_PATH = './data/MF_input_mask.pt'   
x1_PATH = './data/1d_input.pt'
m1_PATH = './data/1d_mask.pt'   
x3_PATH = './data/2d_input.pt'
m3_PATH = './data/2d_mask.pt' 



# Load input data
input_data = torch.load(DATA_PATH)
masks = torch.load(MASK_PATH)
# x1 = torch.load(x1_PATH)
# m1 = torch.load(m1_PATH)
x3 = torch.load(x3_PATH)
m3 = torch.load(m3_PATH)
#print("input data shape: ", input_data.shape)

class MyDataset(Dataset):
    """Custom Dataset for loading the input data and the corresponding masks"""

    def __init__(self, data, masks, x3, m3):
        self.data = data
        self.masks = masks
        self.x3 = x3
        self.m3 = m3

    def __getitem__(self, index):
        return self.data[index], self.masks[index], self.x3[index], self.m3[index]

    def __len__(self):
        return len(self.data)


class VAE(nn.Module):
    """Variational Autoencoder model"""

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(1741, 256) # 1741 -> 256
        self.fc21 = nn.Linear(256, 256)   # 256 -> 256
        self.fc22 = nn.Linear(256, 256)   # 256 -> 256
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 256)    # 256 -> 256
        self.fc4 = nn.Linear(256, 1741)    # 256 -> 1741
        self.leaky_relu = nn.LeakyReLU(0.1)



        self.conv1 = nn.Conv2d(2, 3, kernel_size=17, stride=5, padding=1)    # (n, 3, 15, 15)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)    # (n, 3, 15, 15)
        self.bn1 = nn.BatchNorm2d(3)
        self.flatten_size = 3 * 15 * 15  # Update this based on the new dimensions 867
        
        # Concatenated Encoder
        self.fc5 = nn.Linear(3 * 15 * 15 + 256, 512)
        self.fc6 = nn.Linear(512, 256)
        
        
    
        self.fc7 = nn.Linear(256, 3 * 15 * 15)
        self.fc8 = nn.Linear(3 * 15 * 15, 3 * 15 * 15)
        self.deconv1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(3, 2, kernel_size=17, stride=5, padding=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc21.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc22.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x1, m1, x3, m3):
        
        x1 = x1 * m1
        x3 = x3 * m3
        
        # 1d encode
        x1_res = x1
        x1 = self.dropout(self.leaky_relu(self.fc1(x1)))  # (n, 256)
        
        
        # 2d encode
        x3 = self.leaky_relu(self.conv1(x3))  # (n, 3, 15, 15)
        x3 = self.leaky_relu(self.conv2(x3))  # (n, 3, 15, 15)
        x3 = self.bn1(x3)  # (n, 3, 15, 15)
        x3 = self.dropout(x3)  # (n, 3, 15, 15)
        x3 = x3.view(-1, self.flatten_size)  # (n, 675)
        
        # concate
        #x1 += x1_res
        x = torch.cat((x1, x3), dim=1)  # (n, 931)
        x = self.leaky_relu(self.fc5(x))  # (n, 512)
        x = self.leaky_relu(self.fc6(x))  # (n, 256)
        x = self.dropout(x)  # (n, 256)
        
        # latent space
        mu = self.fc21(x)  # (n, 256)
        log_var = self.fc22(x)  # (n, 256)
        
        # 1d decode
        x1_decoded = self.fc3(x)  # (n, 256)
        x1_decoded = self.fc4(x1_decoded)  # (n, 1741)
        x1_decoded = torch.sigmoid(x1_decoded)  # (n, 1741
        
        # 2d decode
        x3_decoded = self.leaky_relu(self.fc7(x))  # (n, 675)
        x3_decoded = self.leaky_relu(self.fc8(x3_decoded))  # (n, 675)
        x3_decoded = x3_decoded.view(-1, 3, 15, 15)  # (n, 3, 15, 15)
        x3_decoded = self.leaky_relu(self.deconv1(x3_decoded))  # (n, 3, 15, 15)
        x3_decoded = self.deconv2(x3_decoded)  # (n, 2, 75, 75)
        x3_decoded = torch.sigmoid(x3_decoded)  # (n, 2, 75, 75)
        
        return x1_decoded, x3_decoded, mu, log_var



def loss_function(x1_decoded, x3_decoded, x1, x3, m1, m3, mu, logvar):
    # calculate BCE for matrix loss
    x3_decoded_adjacency = x3_decoded[:, 0, :, :] * m3[:, 0, :, :] # shape: [195, 85, 85]
    x3_decoded_structure = x3_decoded[:, 1, :, :] * m3[:, 1, :, :]  # shape: [195, 85, 85]

    x3_adjacency = x3[:, 0, :, :] * m3[:, 0, :, :] # shape: [195, 85, 85]
    x3_structure = x3[:, 1, :, :] * m3[:, 1, :, :] # shape: [195, 85, 85]

    structure_loss = F.binary_cross_entropy(x3_decoded_structure, x3_structure)  
    adjacency_loss = F.binary_cross_entropy(x3_decoded_adjacency, x3_adjacency)

    beta = 1
    gamma = 1
    matrix_loss =  beta * structure_loss + gamma * adjacency_loss

    
    # 1d loss 
    recon_x = x1_decoded
    x = x1
    mask = m1
    seq_loss = F.binary_cross_entropy(recon_x[:, :SEQ_LENGTH], x[:, :SEQ_LENGTH], reduction='none')
    target_loss = F.binary_cross_entropy(recon_x[:, SEQ_LENGTH:1732], x[:, SEQ_LENGTH:1732], reduction='none')
    class_loss = F.binary_cross_entropy(recon_x[:, 1732:1740], x[:, 1732:1740], reduction='none')
    score_loss = F.mse_loss(recon_x[:, -1], x[:, -1], reduction='none')
    
    # apply mask
    seq_loss = seq_loss * mask[:, :SEQ_LENGTH]
    
    # KLD loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    #TODO: change weights of the different losses
    return seq_loss.sum() + target_loss.sum() + class_loss.sum() + score_loss.sum() + KLD + matrix_loss


def train(model, optimizer, epoch, train_loader):
    """Training function for one epoch"""
    model.train()
    train_loss = 0
    for batch_idx, (x1, m1, x3, m3) in enumerate(train_loader):   
        x1 = x1.float().to(device)
        m1 = m1.float().to(device)
        x3 = x3.float().to(device)
        m3 = m3.float().to(device)
        optimizer.zero_grad()
        x1_decoded, x3_decoded, mu, logvar = model(x1, m1, x3, m3)
        #print("other shape: ", recon_batch.shape, data.shape, mu.shape, logvar.shape)
        loss = loss_function(x1_decoded, x3_decoded, x1, x3, m1, m3, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        writer.add_scalar('training_loss', loss.item() / len(x1), epoch * len(train_loader) + batch_idx)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def validate(model, epoch, val_loader):
    """Validation function"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x1, m1, x3, m3) in enumerate(val_loader):
            x1 = x1.float().to(device)
            m1 = m1.float().to(device)
            x3 = x3.float().to(device)
            m3 = m3.float().to(device)
            x1_decoded, x3_decoded, mu, logvar = model(x1, m1, x3, m3)
            val_loss += loss_function(x1_decoded, x3_decoded, x1, x3, m1, m3, mu, logvar).item()
        val_loss /= len(val_loader.dataset)
    writer.add_scalar('validation_loss', val_loss, epoch)
    print('====> Validation set loss: {:.4f}'.format(val_loss))
    return val_loss



def test(epoch, model, test_loader, device):
    """Test function"""
    model.eval()
    test_loss = 0
    total_edit_distance = 0
    total_seqs = 0
    correct_classes = 0
    total_classes = 0
    total_score_error = 0
    y_true = []
    y_pred = []
    class_dict = {0: 'CS', 1: 'ALD', 2: 'BE', 3: 'DCA', 4: 'DIS', 5: 'DOG', 6: 'PRO', 7: 'TES'}
    class_names = ['CS', 'ALD', 'BE', 'DCA', 'DIS', 'DOG', 'PRO', 'TES']

    all_class_indices = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]

    #print(len(test_loader))
    
    for i, (x1, m1, x3, m3) in enumerate(test_loader):
        x1 = x1.float().to(device)
        m1 = m1.float().to(device)
        x3 = x3.float().to(device)
        m3 = m3.float().to(device)
        x1_decoded, x3_decoded, mu, logvar = model(x1, m1, x3, m3)
        test_loss += loss_function(x1_decoded, x3_decoded, x1, x3, m1, m3, mu, logvar).item()
        # combine predicted and actual input into one vector
        data = x1
        recon_batch = x1_decoded
        comparison = [data.view(-1, OUTPUT_SIZE), recon_batch.view(-1, OUTPUT_SIZE)]   
        
        # extract 
        original_seq = onehot_to_seq(comparison[0][0].detach().cpu().numpy()[:SEQ_LENGTH]) # get the first 580 elements for sequence
        original_seq_class = data[0].detach().cpu().numpy()[1732:1740] # get the second last element for class
        original_seq_score = data[0].detach().cpu().numpy()[-1] # get the last element for score
        reconstructed_seq = onehot_to_seq(comparison[1][0].detach().cpu().numpy()[:SEQ_LENGTH]) # get the first 580 elements for sequence
        reconstructed_seq_class = recon_batch[0].detach().cpu().numpy()[1732:1740] # get the second last element for class
        reconstructed_seq_score = recon_batch[0].detach().cpu().numpy()[-1] # get the last element for score
        
        # Compute the mean absolute error of the scores
        score_error = np.abs(reconstructed_seq_score - original_seq_score)
        total_score_error += score_error
        
        # Decode the class back to molecule
        decoded_original_class = decode_class(original_seq_class)
        decoded_reconstructed_class = decode_class(reconstructed_seq_class)
        print('-' * 89)
        print(f'SAMPLE {i+1}')
        print(f'original score: {original_seq_score}, reconstructed score: {reconstructed_seq_score}')
        print(f'original class tensor: {original_seq_class}\nreconstructed class tensor: {reconstructed_seq_class}')
        # Compute the Edit distance and increment the total
        total_edit_distance += compute_edit_distance(original_seq, reconstructed_seq)
        total_seqs += 1
        
        top2_indices = torch.topk(torch.from_numpy(reconstructed_seq_class), 2).indices  # get the top 2 indices
        top2_indices = top2_indices.numpy()
        #top2_indices = torch.topk(torch.from_numpy(reconstructed_seq_class), 2).indices.numpy()
        #print(f"top2: {top2_indices}")
        
        print(f'decoded_original_class: {decoded_original_class[0]}, decoded_reconstructed_class: {decoded_reconstructed_class}')
        # manage all in index first
        true_index = np.argmax(original_seq_class)
        
        #TODO: if want top two, remove the [0] from pred_index
        pred_index = top2_indices[0]
        y_true.append(true_index)
        
        # Top 2 predictions
        if not isinstance(pred_index, (int, np.integer)):
            if true_index in pred_index:
                correct_classes += 1
                y_pred.append(true_index)
            else:
                y_pred.append(pred_index[0])

        # one to one matching
        else:
            if true_index == pred_index:
                correct_classes += 1
                y_pred.append(true_index)
            else:
                y_pred.append(pred_index)

        #print("original class: ", class_dict[true_index], "reconstructed class: ", class_dict[pred_index[0]])
        print("y_true: ", y_true, "y_pred: ", y_pred)
    
    test_loss /= len(test_loader.dataset)
    average_edit_distance = total_edit_distance / total_seqs
    accuracy = correct_classes / total_seqs  # correct_classes is now the number of times the true class was in the top 2 predictions
    average_score_error = total_score_error / total_seqs
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print('-' * 89)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Average Edit distance: {:.4f}'.format(average_edit_distance))
    print('====> Class accuracy: {:.4f}'.format(accuracy))
    print('====> Average score error: {:.4f}'.format(average_score_error))
    print('====> Precision: {:.4f}'.format(precision))
    print('====> Recall: {:.4f}'.format(recall))
    print('====> F1 Score: {:.4f}'.format(f1))
    
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('class_accuracy', accuracy, epoch)
    writer.add_scalar('score_error', average_score_error, epoch)
    writer.add_scalar('edit_distance', average_edit_distance, epoch)
    writer.add_scalar('precision', precision, epoch)
    writer.add_scalar('recall', recall, epoch)
    writer.add_scalar('f1_score', f1, epoch)

    # Uncomment these lines when you run the code in a local environment that supports plots.
    cm = confusion_matrix(y_true, y_pred, labels=all_class_indices)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues')

    #plt.ylabel('Actual label')
    #plt.xlabel('Predicted label')
    plt.title('Confusion Matrix for Steroid Classes', size = 15)

    # Change the tick labels to the class names
    plt.xticks(ticks=np.arange(len(class_names))+0.5, labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names))+0.5, labels=class_names)

    # Remove the tick lines
    plt.tick_params(axis='both', which='both', length=0)

    plt.show()


    # Compute the ROC curve
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=all_class_indices)
    y_pred_bin = label_binarize(y_pred, classes=all_class_indices)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def main():
    """Main function"""
    seed_everything(SEED)
    dataset = MyDataset(input_data, masks, x3, m3)     # pass both data and masks to the dataset

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)   
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True) 

    
    model = VAE().to(device)
    
    # L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in tqdm(range(1, EPOCHS + 1)):
        
        # training model
        train(model, optimizer, epoch, train_loader)
        # validating model
        val_loss = validate(model, epoch, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving model...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_NAME)
            epochs_without_improvement = 0
        elif val_loss >= best_val_loss:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_LIMIT:
            print("Stopping training early.")
            break
        #print(epochs_without_improvement)
        # if epoch % 1000 == 0:
        #     with torch.no_grad():
        #         _, _, _, latent_space = model(input_data.to(device))
        #         plot_t_sne(latent_space, epoch, './latent_space/best_model')
    # testing model
    test(epoch, model, test_loader, device)
    writer.close()



if __name__ == "__main__":
    main()
    
    # Model summary
    # model = VAE(INPUT_SIZE, LAYER_1_SIZE, LAYER_2_SIZE, OUTPUT_SIZE).to(device)
    # summary(model, input_size=(1, INPUT_SIZE))


