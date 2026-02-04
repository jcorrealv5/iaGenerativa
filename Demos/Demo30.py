import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2, os

print("Demo 30: Creando Archivos de Ropas con GAN")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/FashionMNIST/GAN_M4_E1000.pt"
G=torch.jit.load(archivo, map_location=device)
G.eval()

batch_size = 100
ruta = "C:/Data/Python/2026_01_IAG/Demos/datasets/Ropas"
c=0
for i in range(100):
    ruido=torch.randn(batch_size,100).to(device=device)
    imagenesGeneradas=G(ruido).cpu().detach()
    for j in range(batch_size):
        c = c + 1
        print("Creando imagen: " + str(c))
        img = imagenesGeneradas[j]/2+0.5 #-1,1
        img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
        img = img.reshape(28, 28).cpu().numpy() #28x28
        archivo = os.path.join(ruta, str(c) + ".png")
        cv2.imwrite(archivo, img)