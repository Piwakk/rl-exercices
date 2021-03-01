import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import VariationalAutoencoder, VariationalAutoencoderConv, vae_loss
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

latent_dims = 2
EPOCHS = 100
batch_size = 128
capacity = 64
lr = 1e-3
beta = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
conv = False
run_id = f'conv{conv}_latent{latent_dims}_ep{EPOCHS}_lr{lr}_capacity{capacity}_batch{batch_size}'


TRANSFORMS = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=TRANSFORMS, target_transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=TRANSFORMS, target_transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if conv:
	vae = VariationalAutoencoderConv(capacity, latent_dims)
else:
	vae = VariationalAutoencoder(capacity, latent_dims)

vae = vae.to(device)
vae.train()

optim = torch.optim.Adam(params=vae.parameters(), lr=lr, weight_decay=1e-5)

writer = SummaryWriter(f'runs/{run_id}')
PATH = f"models/{run_id}.pt"

for epoch in range(EPOCHS):
    num_batches = 0
    train_loss = 0
    
    for image_batch, _ in train_dataloader:
        
        image_batch = image_batch.to(device)
        if not(conv):
        	image_batch = image_batch.reshape(-1, 784)
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, beta)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    writer.add_scalar('train/loss', train_loss, epoch + 1)
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, EPOCHS, train_loss))
    torch.save(vae, PATH)
