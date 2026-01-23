import torch, torchvision
import numpy as np
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime

def plotearCaras(imagenes):
    figura, ejes = plt.subplots(4,8)
    for i in range(4):
        for j in range(8):
            n = (i * 8) + j
            imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
            ejes[i, j].imshow(imgCara)
    plt.show()

latent_dims=100
class Encoder(nn.Module):
    def __init__(self, latent_dims=100):  
        super().__init__()
        # input 256 by 256 by 3 channels
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        # 128 by 128 with 8 channels
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        # 64 by 64 with 16 channels
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  
        # 31 by 31 with 32 channels
        self.linear1 = nn.Linear(31*31*32, 1024)
        self.linear2 = nn.Linear(1024, latent_dims)
        self.linear3 = nn.Linear(1024, latent_dims)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        std = torch.exp(self.linear3(x))
        z = mu + std*self.N.sample(mu.shape)
        return mu, std, z

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
        x = x.to(device)
        mu, std, z = self.encoder(x)
        return mu, std, self.decoder(z)

print("1. Crear el DataSet y DataLoader de Pruebas con las caras con y sin lentes")
transform = T.Compose([T.Resize(256), T.ToTensor()])
dataset = torchvision.datasets.ImageFolder(root="datasets/LentesTest", transform=transform)
batch_size=16
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
imagenes, etiquetas = next(iter(dataloader))

print("2. Cargar el Modelo VAE Pre-Entrenado con los pesos")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo=VAE().to(device)
modelo.eval()
modelo.load_state_dict(torch.load('preentrenados/VAE/Lentes/VAE_Lentes_487_512.5182352437602.pt', map_location=device))

print("3. Generar Nuevos Rostros y Dibujarlos")
imgs = imagenes.to(device)
mu, std, out = modelo(imgs)
imagenes=torch.cat([imgs[0:4],imgs[4:8],out[0:4],out[4:8],
                  imgs[8:12],imgs[12:16],out[8:12],out[12:16]],
                 dim=0).detach().cpu()
plotearCaras(imagenes)