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
dataset_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
print("dataset_test", dataset_test)

print("4. Crear el DataLoader de Pruebas")
batchSize = 100
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=True)

def plotearDigito(x):
    x = x.detach().cpu().reshape(28, 28)
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

n1 = int(input("Ingresa el Primer Digito a Interpolar: "))
n2 = int(input("Ingresa el Segundo Digito a Interpolar: "))
n = int(input("Ingresa el Numero de muestras generadas: "))

print("5. Obtener los Digitos n1 y n2")
X_test, y_test = next(iter(loader_test))
x1 = X_test[y_test == n1][1].to(device)
x2 = X_test[y_test == n2][1].to(device)
print("x1: ", x1.shape)
print("x2: ", x2.shape)
plotearDigito(x1)
plotearDigito(x2)

print("6. Interpolar entre los digitos 1 y 0")
x1 = x1.view(1, 784).to(device)
mean1, logvar1 = modelo.encode(x1)
z1 = modelo.reparameterization(mean1, logvar1)
x2 = x2.view(1, 784).to(device)
mean2, logvar2 = modelo.encode(x2)
z2 = modelo.reparameterization(mean2, logvar2)
z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
print(z.shape)
listaInterpolada = modelo.decode(z)
arrayInterpolado = listaInterpolada.to('cpu').detach().numpy()
print(arrayInterpolado.shape)

print("7. Graficar la Interpolacion entre los digitos 1 y 0")
w = 28
img = np.zeros((w, n*w))
for i, x_hat in enumerate(arrayInterpolado):
    img[:, i*w:(i+1)*w] = x_hat.reshape(w, w)
plt.imshow(img, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.show()
