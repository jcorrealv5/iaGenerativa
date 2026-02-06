import torch, torchvision
import numpy as np
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime
from modVAE import VAE

def plotearCaras(imagenes, filas, cols):
    figura, ejes = plt.subplots(filas,cols)
    for i in range(filas):
        for j in range(cols):
            n = (i * cols) + j
            imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
            ejes[i, j].imshow(imgCara)
    plt.show()

latent_dims=100
print("Demo 21: Transicion en la Codificacion con VAE")

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

print("5. Calcular el Promedio de Cada Grupo")
mediaHombresSinLentes=codesHombresSinLentes.mean(dim=0)
mediaMujeresSinLentes=codesMujeresSinLentes.mean(dim=0)
mediaHombresConLentes=codesHombresConLentes.mean(dim=0)
mediaMujeresConLentes=codesMujeresConLentes.mean(dim=0)

print("6. Decodificar las Imagenes de Cada Grupo")
print("FUNCIONA POR ESTO: ", mediaHombresSinLentes.unsqueeze(0).shape)
imgMediaHombresSinLentes=modelo.decoder(mediaHombresSinLentes.unsqueeze(0))
imgMediaMujeresSinLentes=modelo.decoder(mediaMujeresSinLentes.unsqueeze(0))
imgMediaHombresConLentes=modelo.decoder(mediaHombresConLentes.unsqueeze(0))
imgMediaMujeresConLentes=modelo.decoder(mediaMujeresConLentes.unsqueeze(0))

imgsMedias = torch.cat([imgMediaHombresSinLentes, imgMediaMujeresSinLentes, imgMediaHombresConLentes, imgMediaMujeresConLentes], dim=0).to(device)
imgsMedias = imgsMedias.squeeze(0).cpu().detach()
plotearCaras(imgsMedias, 2, 2)

def transicionCaras(vector1, vector2):
    results=[]
    for w in np.linspace(0,1,10):
        z=w*vector1+(1-w)*vector2
        out=modelo.decoder(z.unsqueeze(0))
        results.append(out)
    imagenes=torch.cat(results,dim=0).to(device)
    imagenes = imagenes.squeeze(0).cpu().detach()
    plotearCaras(imagenes, 2, 5)

print("7. Transicion de Mujer Con Lentes a Mujer Sin Lentes")
transicionCaras(mediaMujeresSinLentes, mediaMujeresConLentes)

print("8. Transicion de Mujer Sin Lentes a Mujer Con Lentes")
transicionCaras(mediaMujeresConLentes, mediaMujeresSinLentes)

print("9. Transicion de Hombre Con Lentes a Hombre Sin Lentes")
transicionCaras(mediaHombresSinLentes, mediaHombresConLentes)

print("10. Transicion de Hombre Sin Lentes a Hombre Con Lentes")
transicionCaras(mediaHombresConLentes, mediaHombresSinLentes)

print("11. Transicion de Hombre Sin Lentes a Mujer Sin Lentes")
transicionCaras(mediaHombresSinLentes, mediaMujeresSinLentes)

print("12. Transicion de Mujer Sin Lentes a Hombre Sin Lentes")
transicionCaras(mediaMujeresSinLentes, mediaHombresSinLentes)
