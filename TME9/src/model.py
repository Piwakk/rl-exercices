import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderConv(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(EncoderConv, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class DecoderConv(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(DecoderConv, self).__init__()
        c = capacity
        self.capacity = c
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*2, 7, 7)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x
    
class VariationalAutoencoderConv(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(VariationalAutoencoderConv, self).__init__()
        self.encoder = EncoderConv(capacity, latent_dims)
        self.decoder = DecoderConv(capacity, latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Encoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(784, c)
        self.fc_mu = nn.Linear(c, latent_dims)
        self.fc_logvar = nn.Linear(c, latent_dims)
            
    def forward(self, x):
        x = F.relu(self.fc(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Decoder, self).__init__()
        c = capacity
        self.capacity = c
        self.fc_1 = nn.Linear(in_features=latent_dims, out_features=c)
        self.fc_2 = nn.Linear(in_features=c, out_features=784)
            
    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x))
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(capacity, latent_dims)
        self.decoder = Decoder(capacity, latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    

    
def vae_loss(recon_x, x, mu, logvar, variational_beta):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='mean')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + variational_beta * kldivergence
