import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, torchvision
import numpy as np
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime

horaInicio = datetime.now()

def plotearCaras(imagenes):
    figura, ejes = plt.subplots(4,8)
    for i in range(4):
        for j in range(8):
            n = (i * 8) + j
            imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
            ejes[i, j].imshow(imgCara)
    plt.show()

print("1. Crear el DataSet y DataLoader de Entrenamiento con las caras con y sin lentes")
transform = T.Compose([T.Resize(256), T.ToTensor()])
dataset = torchvision.datasets.ImageFolder(root="C:/Users/jhonf/Documents/Shifu/General/datasets/Lentes", transform=transform)
batch_size=32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
imagenes, etiquetas = next(iter(dataloader))
imagen = imagenes[0]
print("Total de Imagenes en el DataSet: ", len(dataset))
print("Shape Imagen: ", imagen.nelement())
plotearCaras(imagenes)

latent_dims=100
class Encoder(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)

        self.linear1 = nn.Linear(31*31*32, 1024)
        self.linear_mu = nn.Linear(1024, latent_dims)
        self.linear_logvar = nn.Linear(1024, latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.linear_mu(x)
        log_var = self.linear_logvar(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return mu, log_var, z


class Decoder(nn.Module):   
    def __init__(self, latent_dims=100):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 31*31*32),
            nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,31,31))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1))
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x  

class VAE(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        # x = x.to(device)
        mu, std, z = self.encoder(x)
        return mu, std, self.decoder(z)

print("2. Crear el Modelo VAE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo=VAE().to(device)
lr=1e-4
optimizador=torch.optim.Adam(modelo.parameters(),lr=lr,weight_decay=1e-5)

print("3. Entrenar el Modelo VAE")
def funcion_perdida(x, x_hat, mean, log_var):
    MSE = F.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return MSE + KLD

def train_epoch(epoch):
    modelo.train()
    perdida_total = 0.0
    nBucles = len(dataloader)
    for i,(imgs, _) in enumerate(dataloader):
        print(f"Item: {i+1}/{nBucles}, Epoca: {epoch+1}")
        imgs = imgs.to(device)
        mu, log_var, out = modelo(imgs)
        perdida = funcion_perdida(imgs, out, mu, log_var)
        perdida_total += perdida.item()
        perdida.backward()
        optimizador.step()
        perdida_promedio = perdida_total/((i+1)*batch_size)
    print(f'Epoca {epoch} - Perdida {perdida_promedio}')
    modelo_cpu = modelo.to("cpu")
    scripted = torch.jit.script(modelo_cpu)
    scripted.save("preentrenados/VAE/Lentes/VAE_Lentes_" + str(epoch+1) + "_" + str(perdida_promedio) + ".pt")
    modelo.to(device)

for i in range(2):
    train_epoch(i)
    with torch.no_grad():
        noise = torch.randn(32,latent_dims).to(device)
        imgs = modelo.decoder(noise).cpu()
        plotearCaras(imgs)

horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo VAE de Caras con Lentes en {tiempoSeg} seg")