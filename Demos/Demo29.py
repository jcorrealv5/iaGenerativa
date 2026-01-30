import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print("Demo 29: Probando la Generacion de Ropas con GAN")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/FashionMNIST/GAN_Ropa_22_6.091745376586914.pt"
G=torch.jit.load(archivo, map_location=device)
G.eval()

batch_size = 32
ruido=torch.randn(batch_size,100).to(device=device)
imagenesGeneradas=G(ruido).cpu().detach()
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow((imagenesGeneradas[i]/2+0.5).reshape(28, 28), cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(hspace=-0.6)
plt.show()