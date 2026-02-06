import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image  # CAMBIO: se usa PIL en lugar de cv2 para guardar imágenes

print("Demo 38: Generando Digitos en Disco usando cGAN")

device = "cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/MNIST/cGAN_Digitos_76_1.1336534023284912.pt"
generador = torch.jit.load(archivo, map_location=device)
generador.eval()

digito = int(input("Ingresa el Digito de 0 a 9 a generar: "))
batch_size = int(input("Cuantos archivos deseas generar con el digito: "))

# CAMBIO: eliminar .cuda() y usar device
z = torch.randn(batch_size, 100, device=device)

# CAMBIO: labels enviados al device sin forzar CUDA
labels = torch.LongTensor([digito for i in range(batch_size)]).to(device)

imagenesGeneradas = generador(z, labels).cpu().detach()

carpeta = str(digito)
if not os.path.isdir(carpeta):
    os.makedirs(carpeta)

c = 0
for i in range(batch_size):
    c += 1
    print("Creando imagen:", c)

    img = imagenesGeneradas[i] / 2 + 0.5        # [-1,1] -> [0,1]
    img = (img.clamp(0, 1) * 255).byte()         # [0,1] -> [0,255]
    img = img.reshape(28, 28).numpy()            # Tensor -> NumPy

    archivo = os.path.join(carpeta, f"{c}.png")
    Image.fromarray(img).save(archivo)           # CAMBIO: guardado con PIL

print(f"Se crearon: {batch_size} archivos con el digito: {digito}")
