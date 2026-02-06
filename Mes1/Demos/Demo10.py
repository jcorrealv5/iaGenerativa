import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Cargar el Modelo PreEntrenado AE")
modelo=torch.jit.load('preentrenados/VAE/MNIST/VAE_Digitos_99_130.20439571940105.pt',map_location=device)
modelo.eval()

print("2. Crear el Transformador para los datos de prueba")
transformacion_data = T.Compose([T.ToTensor()])

print("3. Crear el DataSet de Pruebas")
X_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
print("X_test", X_test)

numEntradas = X_test[0][0].nelement()
print("numEntradas", numEntradas)

print("4. Crear el DataLoader de Pruebas")
batchSize = 100
loader_test = torch.utils.data.DataLoader(X_test, batch_size=batchSize, shuffle=True)

print("5. Mostrar las imagenes originales y las reconstruidas Pre-Entrenadas")

def generarDigito(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = modelo.decode(z_sample)
    digitoGenerado = x_decoded.detach().cpu().reshape(28, 28)
    plt.imshow(digitoGenerado, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit
    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(modelo)
#Plotear el 1 que tiene media 0 y var 1
generarDigito(0.0, 1.0)
#Plotear el 6 que tiene media 0 y var -1
generarDigito(0.0, -1.0)
#Plotear el 8 que tiene media 0.1 y var 0.3
generarDigito(0.1, 0.3)
#Plotear el 0 que tiene media -1.0 y var -1.0
generarDigito(-1.0, -1.0)