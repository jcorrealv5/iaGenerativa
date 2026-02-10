import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("Demo 44: Aritmetica en el Espacio Latente")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/Lentes/cGAN_Lentes_100_1077.6683349609375.pt"
generador=torch.jit.load(archivo, map_location=device)
generador.eval()

def generarRuidoEtiqueta(indice):
    noise_g = torch.randn(1, 100, 1, 1)
    labels_g = torch.zeros(1, 2, 1, 1)
    labels_g[:,indice,:,:]=1
    noise_and_labels=torch.cat([noise_g,labels_g],dim=1).to(device)
    imagenesGeneradas=generador(noise_and_labels)
    return imagenesGeneradas[0],noise_g, labels_g

imgConLentes,ruidoConLentes,labelConLentes = generarRuidoEtiqueta(0)
imgSinLentes,ruidoSinLentes,labelSinLentes = generarRuidoEtiqueta(1)
imgPersona,ruidoPersona,labelPersona = generarRuidoEtiqueta(1)
imgs = [imgConLentes, imgSinLentes, imgPersona]
titulos = ["Con Lentes", "Sin Lentes", "Referencia"]
plt.figure(figsize=(20,4),dpi=50)
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    img=(imgs[i].cpu().detach()/2+0.5).permute(1,2,0)
    plt.xticks([])
    plt.yticks([])
    plt.title(titulos[i])
    plt.imshow(img)
plt.show()

pesos = np.linspace(0,1,6)
nPesos = len(pesos)
plt.figure(figsize=(20,4),dpi=50)
for i in range(nPesos):
    ax = plt.subplot(1, nPesos, i + 1)
    label=pesos[i]*labelSinLentes+(1-pesos[i])*labelConLentes
    noise_and_labels=torch.cat([ruidoPersona.reshape(1, 100, 1, 1),label.reshape(1, 2, 1, 1)],dim=1).to(device) 
    fake=generador(noise_and_labels).cpu().detach()
    img=(fake[0]/2+0.5).permute(1,2,0)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
plt.title("Con Lentes a Sin Lentes")
plt.show()

plt.figure(figsize=(20,4),dpi=50)
for i in range(nPesos):
    ax = plt.subplot(1, nPesos, i + 1)
    label=pesos[i]*labelConLentes+(1-pesos[i])*labelSinLentes
    noise_and_labels=torch.cat([ruidoPersona.reshape(1, 100, 1, 1),label.reshape(1, 2, 1, 1)],dim=1).to(device) 
    fake=generador(noise_and_labels).cpu().detach()
    img=(fake[0]/2+0.5).permute(1,2,0)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
plt.title("Sin Lentes a Con Lentes")
plt.show()