
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELOS = BASE_DIR / "preentrenados" / "GAN" / "MNIST"
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)


horaInicio = datetime.now()

def plotearImagenes(imagenes, filas, cols):
    figura, ejes = plt.subplots(filas,cols)
    for i in range(filas):
        for j in range(cols):
            n = (i * cols) + j
            imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
            ejes[i, j].imshow(imgCara)
    plt.show()

print("Entrenamiento de una GAN para Generar Digitos de MNIST")

print("1. Crear el Transformador para los datos")
transformacion_data = T.Compose([T.ToTensor()])

print("2. Crear el DataSet y DataLoader de Entrenamiento con MNIST")
X_train = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=transformacion_data)
print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
batch_size = 32
loader_train = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

print("3. Cargar un lote de imagenes y plotearlas")
imagenes, etiquetas = next(iter(loader_train))
plotearImagenes(imagenes, 4, 8)

device="cuda" if torch.cuda.is_available() else "cpu"
print("4. Crear el Discriminador")
D=nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()).to(device)

print("4. Crear el Generador")
G=nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 784),
    nn.Tanh()).to(device)

print("5. Definir las Funciones de Perdida y Optimizacion")
funcionPerdida=nn.BCELoss()
lr=0.0001
D_optimizador=torch.optim.Adam(D.parameters(),lr=lr)
G_optimizador=torch.optim.Adam(G.parameters(),lr=lr)

print("6. Definir la Funcion de Entrenamiento del Discriminador con Imagenes Reales")
def train_D_Reales(imagenesReales):
    imagenesReales=imagenesReales.reshape(-1,28*28).to(device)
    D_salida=D(imagenesReales)    
    etiquetas=torch.ones((imagenesReales.shape[0],1)).to(device)
    D_perdida=funcionPerdida(D_salida,etiquetas)    
    D_optimizador.zero_grad()
    D_perdida.backward()
    D_optimizador.step()
    return D_perdida

print("7. Definir la Funcion de Entrenamiento del Discriminador con Imagenes Falsas")
def train_D_Falsas():
    ruido=torch.randn(batch_size,100).to(device=device)
    imagenesFalsas=G(ruido)
    D_salida=D(imagenesFalsas)    
    etiquetas=torch.zeros((batch_size,1)).to(device)
    D_perdida=funcionPerdida(D_salida,etiquetas)    
    D_optimizador.zero_grad()
    D_perdida.backward()
    D_optimizador.step()
    return D_perdida

print("8. Definir la Funcion de Entrenamiento del Generador")
def train_G(): 
    ruido=torch.randn(batch_size,100).to(device=device)
    imagenesGeneradas=G(ruido)
    G_salida=D(imagenesGeneradas)
    etiquetas = torch.ones((batch_size,1)).to(device)
    G_perdida=funcionPerdida(G_salida,etiquetas)
    G_optimizador.zero_grad()
    G_perdida.backward()
    G_optimizador.step()
    return G_perdida

print("9. Definir la Funcion para Plotear lo creado por el Generador")
def plotearGenerados():
    ruido=torch.randn(batch_size,100).to(device=device)
    imagenesGeneradas=G(ruido).cpu().detach()
    plt.figure(dpi=100,figsize=(20,10))
    for i in range(batch_size):
        ax=plt.subplot(4, 8, i + 1)
        imagenes=(imagenesGeneradas[i]/2+0.5).reshape(28, 28)
        plt.imshow(imagenes, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()

print("9. Realizar el Entrenamiento del Discriminador y el Generador")
nMuestras = len(loader_train)
for i in range(200):
    G_perdida_total=0
    D_perdida_total=0
    for n, (imagenes,_) in enumerate(loader_train):
        print(f"item: {n+1} - bucles: {nMuestras} - epoca: {i+1}")
        D_perdida=train_D_Reales(imagenes)
        D_perdida_total+=D_perdida
        D_perdida=train_D_Falsas()
        D_perdida_total+=D_perdida
        G_perdida=train_G()
        G_perdida_total+=G_perdida
    G_perdida_promedio=G_perdida_total/n
    D_perdida_promedio=D_perdida_total/n    
    print(f"Epoca {i+1}, Perdida Discriminador: {D_perdida_promedio}, Perdida Generador: {G_perdida_promedio}")
    scripted = torch.jit.script(G)
    archivo = RUTA_MODELOS / f"GAN_Digitos_{i+1}_{G_perdida_promedio.item():.4f}.pt"
    scripted.save(str(archivo))


plotearGenerados()

horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo VAE de Caras con Lentes en {tiempoSeg} seg")