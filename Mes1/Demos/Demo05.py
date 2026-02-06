import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

horaInicio = datetime.now()

print("1. Crear el Transformador para los datos")
transform = T.Compose([T.ToTensor(),])

print("2. Crear los DataSets y DataLoader de Entrenamiento")
X_train = torchvision.datasets.ImageFolder(root="datasets/Alumnos", transform=transform)
batch_size=16
loader_train = torch.utils.data.DataLoader(X_train, batch_size=batch_size,shuffle=True)

nEntradas = 3*100*100
nOcultas = 1000
nCodigo = 100
class AutoCodificador(nn.Module):
    def __init__(self, nEntradas, nCodigo, nOcultas):
        super().__init__()
        self.inicio = nn.Linear(nEntradas, nOcultas)
        self.codifica = nn.Linear(nOcultas, nCodigo)
        self.norma = nn.Linear(nCodigo, nOcultas)
        self.decodifica = nn.Linear(nOcultas, nEntradas)

    def codificar(self, x):
        x = F.relu(self.inicio(x))
        mu = F.relu(self.codifica(x))
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

print("3. Crear el Modelo con el AutoCodificador")
modelo = AutoCodificador(nEntradas, nCodigo, nOcultas).to(device)

print("4. Crear el Optimizador para el Entrenamiento")
optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.01)

print("5. Mostrar las imagenes originales y las reconstruidas sin Entrenar el AE")
originales = []
indice = 0
for img, etiqueta in X_train:   
    if(etiqueta==indice):
        originales.append(img)
        indice += 1
    if(indice==5):
        break

def plotearImagenes():
    reconstruidas = []
    for i in range(5):
        imgOriginal = originales[i].reshape((1,30000))
        imgReconstruida, mu = modelo(imgOriginal.to(device))
        reconstruidas.append(imgReconstruida.reshape((100,100,3)))
    imagenes = originales + reconstruidas
    print("Shape originales: ", imagenes[0].shape)
    print("Shape reconstruidas: ", imagenes[5].shape)

    figura, ejes = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            n = (i * 5) + j
            if(n<5):
                ejes[i,j].imshow(np.transpose(imagenes[n],(1,2,0)), cmap="gray")
            else:
                ejes[i,j].imshow(imagenes[n].detach().cpu().numpy(), cmap="gray")
    plt.show()

plotearImagenes()

nMuestras = len(loader_train)
print("6. Entrenar el Modelo AE y Mostrar cada Epoca")
criterio = nn.MSELoss()
for epoca in range(100):
    perdida_total = 0   
    for i,(imgs, etiquetas) in enumerate(loader_train):
        print(f"Item: {i+1}/{nMuestras}/{epoca}")
        imgOriginal = imgs.to(device).view(-1, numEntradas)
        imgReconstruida, mu = modelo(imgOriginal)
        perdida = criterio(imgReconstruida, imgOriginal)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
    print(f"Epoca: {epoca} - Perdida: {perdida.item()}")
    if(epoca % 5 == 0):
        scripted = torch.jit.script(modelo)
        scripted.save("preentrenados/AE_Caras_" + str(epoca+1) + "_" + str(perdida.item()) + ".pt")
    plotearImagenes()

horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo AE de Caras en {tiempoSeg} seg")