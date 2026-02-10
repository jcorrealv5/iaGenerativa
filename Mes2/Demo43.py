import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("Demo 43: Probando la Generacion de Caras Con y Sin Lentes con cGAN")
device="cuda" if torch.cuda.is_available() else "cpu"

#cGAN_Alumnos_3000_70.77812194824219
archivo = "preentrenados/GAN/Lentes/cGAN_Lentes_100_1077.6683349609375.pt"
generador=torch.jit.load(archivo, map_location=device)
generador.eval()

batch_size = 12

def plotearCaras(imagenesGeneradas, titulo):
    for i in range(batch_size):
        ax = plt.subplot(2, 6, i + 1)
        img = imagenesGeneradas[i]/2+0.5 #-1,1
        img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
        img = img.permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.title(titulo)
    plt.subplots_adjust(hspace=-0.6)
    plt.show()

def generarCaras(indice, titulo):
    noise_g = torch.randn(batch_size, 100, 1, 1)
    labels_g = torch.zeros(batch_size, 2, 1, 1)
    labels_g[:,indice,:,:]=1
    noise_and_labels=torch.cat([noise_g,labels_g],dim=1).to(device)
    imagenesGeneradas=generador(noise_and_labels)
    plotearCaras(imagenesGeneradas, titulo)

generarCaras(0, "Con Lentes")
generarCaras(1, "Sin Lentes")