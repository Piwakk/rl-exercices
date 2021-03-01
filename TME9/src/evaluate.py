import torch
from model import vae_loss
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reconstruction_error(dataloader, model, beta):
    
    loss, num_batches = 0, 0
    for image_batch, _ in dataloader:
        
        with torch.no_grad():
        
            image_batch = image_batch.to(device)
            image_batch_recon, latent_mu, latent_logvar = model(image_batch)
            loss += vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, beta).item()
            num_batches += 1
        
    loss /= num_batches
    print('average reconstruction error: %f' % (loss))
    return loss

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):
    with torch.no_grad():
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

def reconstruction(dataloader, model):
    plt.ion()
    images, labels = iter(dataloader).next()
    print('Original images')
    show_image(torchvision.utils.make_grid(images[1:50],10,5))
    plt.show()
    print('VAE reconstruction:')
    visualise_output(images, model)

def interpolation(rho, model, x1, x2):
    with torch.no_grad():
        x1 = x1.to(device)
        h1, _ = model.encoder(x1)
        x2 = x2.to(device)
        h2, _ = model.encoder(x2)
        h = rho * h1 + (1- rho) * h2
        x = model.decoder(h)
        x = x.cpu()
        return x

def interpolation_plot(model, digits):
    lambda_range=np.linspace(0,1,10)
    fig, axs = plt.subplots(2,5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for ind,l in enumerate(lambda_range):
        inter_image=interpolation(float(l), model, digits[7][0], digits[1][0])
        inter_image = to_img(inter_image)
        image = inter_image.numpy()
        axs[ind].imshow(image[0,0,:,:], cmap='gray')
        axs[ind].set_title('lambda_val='+str(round(l,1)))
    plt.show() 

def sort_digit(loader):
    digits = [[] for _ in range(10)]
    for img_batch, label_batch in loader:
        for i in range(img_batch.size(0)):
            digits[label_batch[i]].append(img_batch[i:i+1])
        if sum(len(d) for d in digits) >= 1000:
            break;
    return digits

def generate(model, latent_dims):
    with torch.no_grad():
        latent = torch.randn(128, latent_dims, device=device)
        img_recon = model.decoder(latent)
        img_recon = img_recon.cpu()
        fig, ax = plt.subplots(figsize=(5, 5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100],10,5))
        plt.show()

def generation_map(model):
    with torch.no_grad():
        latent_x = np.linspace(-1.5,1.5,20)
        latent_y = np.linspace(-1.5,1.5,20)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), 2)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2)
        latents = latents.to(device)
        image_recon = model.decoder(latents)
        image_recon = image_recon.cpu()
        fig, ax = plt.subplots(figsize=(10, 10))
        show_image(torchvision.utils.make_grid(image_recon.data[:400],20,5))
        plt.show()

if __name__=='__main__':

    PATH = 'models/latent2_ep100_lr0.001_capacity64_batch128_beta1e-05_1614553086_894832.pt'
    vae = torch.load(PATH, map_location='cpu')
    vae.eval()

    batch_size = 128
    beta = 1e-5
    latent_dims = 2
    TRANSFORMS = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=TRANSFORMS, target_transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    reconstruction_error(test_dataloader, vae, beta)
    reconstruction(test_dataloader, vae)
    interpolation_plot(vae, sort_digit(test_dataloader))
    generate(vae)
    generation_map(vae)


