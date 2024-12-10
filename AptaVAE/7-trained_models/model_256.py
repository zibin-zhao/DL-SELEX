import torch
import torch.nn as nn
    
    
class VAE(nn.Module):
    """Variational Autoencoder model"""

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(1645, 256) # 1741 -> 256
        self.fc41 = nn.Linear(256, 256)   # 256 -> 256
        self.fc42 = nn.Linear(256, 256)   # 256 -> 256
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(256, 256)    # 256 -> 256
        self.fc6 = nn.Linear(256, 1645)    # 256 -> 1741
        self.leaky_relu = nn.LeakyReLU(0.1)

        # input conv (188, 3, 100, 100)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=6, stride=4, padding=1)    # (n, 3, 25, 25)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)    # (n, 3, 25, 25)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.flatten_size = 3 * 25 * 25  # Update this based on the new dimensions 867
        
        # Concatenated Encoder
        self.fc2 = nn.Linear(self.flatten_size + 256, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Decoder
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, self.flatten_size)
        self.deconv1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(3, 3, kernel_size=6, stride=4, padding=1)
        
        # Kaiming Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # For 1d decoding purpose only
    def decode(self, z):
        x1_decoded = self.fc5(z)  # (n, 256)
        x1_decoded = self.fc6(x1_decoded)  # (n, 1741)
        x1_decoded = torch.sigmoid(x1_decoded)  # (n, 1741)
        return x1_decoded
    
    def forward(self, x1, m1, x3, m3):
        
        x1 = x1 * m1 
        x3 = x3 * m3
        
        # 1d encode
        #x1_res = x1
        x1 = self.dropout(self.leaky_relu(self.fc1(x1)))  # (n, 256)
        
        # 2d encode
        # x3 = self.conv1(self.leaky_relu(self.bn1(x3)))  
        # x3 = self.conv2(self.leaky_relu(self.bn2(x3)))  
        x3 = self.leaky_relu(self.conv1(x3))  # (n, 3, 15, 15)
        x3 = self.leaky_relu(self.conv2(x3))  # (n, 3, 15, 15)
        x3 = self.bn1(x3)  # (n, 3, 15, 15)
        x3 = self.dropout(x3)  # (n, 3, 15, 15)
        x3 = x3.view(-1, self.flatten_size)  # (n, 675)
    
        # concate
        #x1 += x1_res
        x = torch.cat((x1, x3), dim=1)  # (n, 931)
        x = self.leaky_relu(self.fc2(x))  # (n, 512)
        x = self.leaky_relu(self.fc3(x))  # (n, 256)
        x += x1     # res connection for fc1 to fc3
        x = self.dropout(x)  # (n, 256)
        
        
        # latent space
        mu = self.fc41(x)  # (n, 256)
        log_var = self.fc42(x)  # (n, 256)
        z = self.reparameterize(mu, log_var)
        
        # 1d decode
        x1_decoded = self.decode(z)
        
        # 2d decode
        x3_decoded = self.leaky_relu(self.fc7(z))  # (n, 675)
        x3_decoded = self.leaky_relu(self.fc8(x3_decoded))  # (n, 675)
        x3_decoded = x3_decoded.view(-1, 3, 25, 25)  # (n, 3, 15, 15)
        x3_decoded = self.leaky_relu(self.deconv1(x3_decoded))  # (n, 3, 15, 15)
        x3_decoded = self.deconv2(x3_decoded)  # (n, 2, 75, 75)
        x3_decoded = torch.sigmoid(x3_decoded)  # (n, 2, 75, 75)
        
        return x1_decoded, x3_decoded, mu, log_var, z

    def encode_to_latent(self, x1, m1, x3, m3):
        _, _, _, _, z = self.forward(x1, m1, x3, m3)
        return z.detach().numpy()