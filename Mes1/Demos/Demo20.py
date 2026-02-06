import torch, torchvision
import numpy as np
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime

def plotearCaras(imagenes, filas, cols):
    figura, ejes = plt.subplots(filas,cols)
    for i in range(filas):
        for j in range(cols):
            n = (i * cols) + j
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

print("Demo 20: Aritmetica de Codificacion con VAE")

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

print("3. Seleccionar 4 Grupos de Imagenes")
imgs = imagenes.to(device)
mu, std, out = modelo(imgs)
imgsHombresSinLentes=torch.cat([imgs[0].unsqueeze(0), imgs[1].unsqueeze(0), imgs[2].unsqueeze(0), imgs[3].unsqueeze(0)], dim=0).to(device)
imgsMujeresSinLentes=torch.cat([imgs[4].unsqueeze(0), imgs[5].unsqueeze(0), imgs[6].unsqueeze(0), imgs[7].unsqueeze(0)], dim=0).to(device)
imgsHombresConLentes=torch.cat([imgs[8].unsqueeze(0), imgs[9].unsqueeze(0), imgs[10].unsqueeze(0), imgs[11].unsqueeze(0)], dim=0).to(device)
imgsMujeresConLentes=torch.cat([imgs[12].unsqueeze(0), imgs[13].unsqueeze(0), imgs[14].unsqueeze(0), imgs[15].unsqueeze(0)], dim=0).to(device)
print("imgsHombresSinLentes: ", imgsHombresSinLentes.shape)

print("4. Codificar los Grupos")
_,_,codesHombresSinLentes=modelo.encoder(imgsHombresSinLentes)
_,_,codesMujeresSinLentes=modelo.encoder(imgsMujeresSinLentes)
_,_,codesHombresConLentes=modelo.encoder(imgsHombresConLentes)
_,_,codesMujeresConLentes=modelo.encoder(imgsMujeresConLentes)
print("codesHombresSinLentes Shape: ", codesHombresSinLentes.shape)
print("codesHombresSinLentes: ", codesHombresSinLentes)

print("5. Calcular el Promedio de Cada Grupo")
mediaHombresSinLentes=codesHombresSinLentes.mean(dim=0)
mediaMujeresSinLentes=codesMujeresSinLentes.mean(dim=0)
mediaHombresConLentes=codesHombresConLentes.mean(dim=0)
mediaMujeresConLentes=codesMujeresConLentes.mean(dim=0)
print("mediaHombresSinLentes Shape: ", mediaHombresSinLentes.shape)
print("mediaHombresSinLentes: ", mediaHombresSinLentes)
#print("mediaMujeresSinLentes: ", mediaMujeresSinLentes)
#print("mediaHombresConLentes: ", mediaHombresConLentes)
#print("mediaMujeresConLentes: ", mediaMujeresConLentes)

print("6. Decodificar las Imagenes de Cada Grupo")
imgMediaHombresSinLentes=modelo.decoder(mediaHombresSinLentes.unsqueeze(0))
imgMediaMujeresSinLentes=modelo.decoder(mediaMujeresSinLentes.unsqueeze(0))
imgMediaHombresConLentes=modelo.decoder(mediaHombresConLentes.unsqueeze(0))
imgMediaMujeresConLentes=modelo.decoder(mediaMujeresConLentes.unsqueeze(0))

imgsMedias = torch.cat([imgMediaHombresSinLentes, imgMediaMujeresSinLentes, imgMediaHombresConLentes, imgMediaMujeresConLentes], dim=0).to(device)
imgsMedias = imgsMedias.squeeze(0).cpu().detach()
plotearCaras(imgsMedias, 2, 2)

def distanciaEuclidiana(x):
    d1 = torch.linalg.norm(mediaHombresSinLentes - x, ord=2).item()
    d2 = torch.linalg.norm(mediaMujeresSinLentes - x, ord=2).item()
    d3 = torch.linalg.norm(mediaHombresConLentes - x, ord=2).item()
    d4 = torch.linalg.norm(mediaMujeresConLentes - x, ord=2).item()
    distancias = np.array([d1,d2,d3,d4])
    indice = np.argmin(distancias)
    return indice, distancias[indice]

zMediaHombreConLentes = mediaHombresSinLentes - mediaMujeresSinLentes + mediaMujeresConLentes
out=modelo.decoder(zMediaHombreConLentes.unsqueeze(0))
imgsAritmetica1 = torch.cat([imgMediaHombresSinLentes, imgMediaMujeresSinLentes, out, imgMediaMujeresConLentes], dim=0).to(device)
imgsAritmetica1 = imgsAritmetica1.squeeze(0).cpu().detach()
indice, distancia = distanciaEuclidiana(zMediaHombreConLentes) #2
print(f"El Vector Mas Cercano a HCL es: {indice} con distancia: {distancia}")
plotearCaras(imgsAritmetica1, 2, 2)

zMediaMujerConLentes = mediaMujeresSinLentes - mediaHombresSinLentes + mediaHombresConLentes
out=modelo.decoder(zMediaMujerConLentes.unsqueeze(0))
imgsAritmetica2 = torch.cat([imgMediaHombresSinLentes, imgMediaMujeresSinLentes, imgMediaHombresConLentes, out], dim=0).to(device)
imgsAritmetica2 = imgsAritmetica2.squeeze(0).cpu().detach()
indice, distancia = distanciaEuclidiana(zMediaMujerConLentes) #3
print(f"El Vector Mas Cercano a MCL es: {indice} con distancia: {distancia}")
plotearCaras(imgsAritmetica2, 2, 2)

zMediaHombreSinLentes = mediaHombresConLentes - mediaMujeresConLentes + mediaMujeresSinLentes
out=modelo.decoder(zMediaHombreSinLentes.unsqueeze(0))
imgsAritmetica3 = torch.cat([out, imgMediaMujeresSinLentes, imgMediaHombresConLentes, imgMediaMujeresConLentes], dim=0).to(device)
imgsAritmetica3 = imgsAritmetica3.squeeze(0).cpu().detach()
indice, distancia = distanciaEuclidiana(zMediaHombreSinLentes)
print(f"El Vector Mas Cercano a HSL es: {indice} con distancia: {distancia}")
plotearCaras(imgsAritmetica3, 2, 2) #0

zMediaMujerSinLentes = mediaHombresSinLentes - mediaHombresConLentes + mediaMujeresConLentes
out=modelo.decoder(zMediaMujerSinLentes.unsqueeze(0))
imgsAritmetica4 = torch.cat([imgMediaHombresSinLentes, out, imgMediaHombresConLentes, imgMediaMujeresConLentes], dim=0).to(device)
imgsAritmetica4 = imgsAritmetica4.squeeze(0).cpu().detach()
indice, distancia = distanciaEuclidiana(zMediaMujerSinLentes)
print(f"El Vector Mas Cercano a MSL es: {indice} con distancia: {distancia}")
plotearCaras(imgsAritmetica4, 2, 2) #1
