import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

print("Demo 38: Generando Digitos en Disco usando cGAN")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/MNIST/cGAN_Digitos_91_0.9423668384552002.pt"
generador=torch.jit.load(archivo, map_location=device)
generador.eval()

digito = int(input("Ingresa el Digito de 0 a 9 a generar: "))
batch_size = int(input("Cuantos archivos deseas generar con el digito: "))

z = Variable(torch.randn(batch_size, 100)).cuda()
labels = torch.LongTensor([digito for i in range(batch_size)]).cuda()
imagenesGeneradas = generador(z, labels).cpu().detach()

carpeta = str(digito)
if(not os.path.isdir(carpeta)):
    os.makedirs(carpeta)
c=0
for i in range(batch_size):
    c = c + 1
    print("Creando imagen: " + str(c))
    img = imagenesGeneradas[i]/2+0.5 #-1,1
    img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
    img = img.reshape(28, 28).cpu().numpy() #28x28
    archivo = os.path.join(carpeta, str(c) + ".png")
    cv2.imwrite(archivo, img)
print(f"Se crearon: {batch_size} archivos con el digito: {digito}")