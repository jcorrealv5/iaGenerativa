import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("1. Crear el Transformador para los datos")
transformacion_data = T.Compose([T.ToTensor()])

print("2. Crear los DataSets de Entrenamiento y Pruebas")
X_train = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=transformacion_data)
print("X_train", X_train)
X_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
print("X_test", X_test)

print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
batchSize = 32
loader_train = torch.utils.data.DataLoader(X_train, batch_size=batchSize, shuffle=True)
loader_test = torch.utils.data.DataLoader(X_test, batch_size=batchSize, shuffle=True)

class AutoCodificador(nn.Module):
    def __init__(self, nEntradas, nCodigo, nOcultas):
        super().__init__()
        self.inicio = nn.Linear(nEntradas, nOcultas)
        self.codifica = nn.Linear(nOcultas, nCodigo)
        self.norma = nn.Linear(nCodigo, nOcultas)
        self.decodifica = nn.Linear(nOcultas, nEntradas)

    def codificar(self, x):
        x = F.relu(self.inicio(x))
        mu = self.codifica(x)
        return mu

    def decodificar(self, z):
        y = F.relu(self.norma(z))
        y = torch.sigmoid(self.decodifica(y))
        return y

    def forward(self, x):
        mu = self.codificar(x)
        y = self.decodificar(mu)
        return y, mu

device = "cuda" if torch.cuda.is_available() else "cpu"
numEntradas = X_train[0][0].nelement()
print("numEntradas", numEntradas)
numCodigo = 20
numOcultas = 200

print("4. Crear el Modelo con el AutoCodificador")
modelo = AutoCodificador(numEntradas, numCodigo, numOcultas)

print("5. Crear el Optimizador para el Entrenamiento")
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.00025)

print("6. Mostrar las imagenes originales y las reconstruidas sin Entrenar el AE")
originales = []
indice = 0
for img, etiqueta in X_test:
    if(etiqueta==indice):
        originales.append(img)
        indice += 1
    if(indice==10):
        break

def plotearImagenes():
    reconstruidas = []
    for i in range(10):
        imgOriginal = originales[i].reshape((1,numEntradas))
        imgReconstruida, mu = modelo(imgOriginal.to(device))
        reconstruidas.append(imgReconstruida)
    imagenes = originales + reconstruidas

    plt.figure(figsize=(10,2),dpi=50)
    for i in range(20):
        ax = plt.subplot(2, 10, i + 1)
        img = imagenes[i].detach().cpu().numpy().reshape(28,28)
        plt.imshow(img, cmap="binary")
    plt.show()

plotearImagenes()

nMuestras = len(loader_train)
print("7. Entrenar el Modelo AE y Mostrar cada Epoca")
for epoca in range(100):
    perdida_total = 0   
    for i,(imgs, etiquetas) in enumerate(loader_train):
        print(f"Item: {i+1}/{nMuestras}/{epoca}")
        imgOriginal = imgs.to(device).view(-1, numEntradas)
        imgReconstruida, mu = modelo(imgOriginal)
        perdida = ((imgReconstruida - imgOriginal)**2).sum()
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()
        perdida_valor = perdida_total / nMuestras
    print(f"Epoca: {epoca} - Perdida: {perdida_valor}")
    if(perdida_valor<180):
        scripted = torch.jit.script(modelo)
        scripted.save("preentrenados/AE_Digitos_" + str(perdida_valor) + ".pt")
    #plotearImagenes()