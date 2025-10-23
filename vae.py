import torch
import torch.nn as nn
import torch.nn.functional as F
# VAE Encoder
class VAEEncoder(nn.Module):
    def __init__(self, num_joints, z_dim=32):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(num_joints * num_joints * 2, 512)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, z_dim=32):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 1)  # 用于解码潜在变量

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x

class VAE(nn.Module):
    def __init__(self, num_joints, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(num_joints, z_dim)
        self.decoder = VAEDecoder(z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # 重参数化技巧

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar