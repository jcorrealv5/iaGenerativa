import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print("Demo 32: Probando la Generacion de Caras de Animes con DC-GAN")
device="cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

archivo = "preentrenados/GAN/Animes/GAN_Animes_100_7.748009204864502.pt"
G=torch.jit.load(archivo, map_location=device)
G.eval()

batch_size = 32
ruido=torch.randn(batch_size,100,1,1).to(device=device)
imagenesGeneradas=G(ruido).cpu().detach()
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow((imagenesGeneradas[i]/2+0.5).permute(1,2,0))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(hspace=-0.6)
plt.show()