import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os  # CAMBIO: manejo correcto de rutas

print("Demo 37: Probando la Generacion de Digitos cGAN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# CAMBIO: ruta correcta porque preentrenados está dentro de Demo37
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(
    BASE_DIR,
    "preentrenados/GAN/MNIST/cGAN_Digitos_68_1.295311450958252.pt"
)

# Cargar modelo
generador = torch.jit.load(archivo, map_location=device)
generador.eval()

batch_size = 100

# CAMBIO: eliminar .cuda() y usar device
z = torch.randn(batch_size, 100, device=device)

# CAMBIO: labels al device correcto
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)

imagenesGeneradas = generador(z, labels).cpu().detach()

for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    plt.imshow((imagenesGeneradas[i] / 2 + 0.5).reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(hspace=-0.6)
plt.show()
