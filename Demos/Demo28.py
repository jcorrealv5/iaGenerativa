import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

horaInicio = datetime.now()

def plotearImagenes(imagenes, filas, cols):
    figura, ejes = plt.subplots(filas,cols)
    for i in range(filas):
        for j in range(cols):
            n = (i * cols) + j
            imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
            ejes[i, j].imshow(imgCara)
    plt.show()

print("Demo 28: Entrenamiento de una GAN para Generar Ropa de Fashion-MNIST")

print("1. Crear el Transformador para los datos")
transformacion_data = T.Compose([T.ToTensor(), T.Normalize([0.5],[0.5])])

print("2. Crear el DataSet y DataLoader de Entrenamiento con MNIST")
X_train = torchvision.datasets.FashionMNIST(root="datasets", train=True, download=True, transform=transformacion_data)
print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
batch_size = 64
epocas = 100
loader_train = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

print("3. Cargar un lote de imagenes y plotearlas")
imagenes, etiquetas = next(iter(loader_train))
plotearImagenes(imagenes, 4, 8)

device="cuda" if torch.cuda.is_available() else "cpu"
print("4. Crear el Discriminador")
D=nn.Sequential(
    nn.Linear(784, 1024),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
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
lr=0.00001
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
    return D_perdida, D_salida

print("7. Definir la Funcion de Entrenamiento del Discriminador con Imagenes Falsas")
def train_D_Falsas():
    ruido=torch.randn(batch_size,100).to(device=device)
    imagenesFalsas=G(ruido).detach()
    D_salida=D(imagenesFalsas)
    etiquetas=torch.zeros((batch_size,1)).to(device)
    D_perdida=funcionPerdida(D_salida,etiquetas)    
    D_optimizador.zero_grad()
    D_perdida.backward()
    D_optimizador.step()
    return D_perdida, D_salida

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
        ax=plt.subplot(8, 8, i + 1)
        imagenes=(imagenesGeneradas[i]/2+0.5).reshape(28, 28)
        plt.imshow(imagenes, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()

print("9. Realizar el Entrenamiento del Discriminador y el Generador")
nMuestras = len(loader_train)
perdidas = []
minPerdida = 1000000
minEpoca = 0
def plotearPerdidas():
    figura = plt.figure()
    ejes = figura.subplots()
    ejes.plot(range(1,epocas+1), perdidas)
    ejes.set_title("Perdidas Promedio del Entrenamiento x Epocas")
    plt.show()

for i in range(epocas):
    G_perdida_total=0
    D_perdida_total=0
    for n, (imagenes,_) in enumerate(loader_train):
        print(f"item: {n+1} - bucles: {nMuestras} - epoca: {i+1}")
        D_perdida,D_salida_real=train_D_Reales(imagenes)
        D_perdida_total+=D_perdida
        D_perdida,D_salida_falsa =train_D_Falsas()
        D_perdida_total+=D_perdida
        G_perdida=train_G()
        G_perdida_total+=G_perdida
    G_perdida_promedio=G_perdida_total/n
    D_perdida_promedio=D_perdida_total/n
    G_perdida_valor = G_perdida_promedio.item()
    perdidas.append(G_perdida_valor)
    if(G_perdida_valor<minPerdida):
        minPerdida = G_perdida_valor
        minEpoca = (i+1)
    print(f"Epoca {i+1}, Perdida Discriminador: {D_perdida_promedio}, Perdida Generador: {G_perdida_promedio}")
    scripted = torch.jit.script(G)
    archivo = "preentrenados/GAN/FashionMNIST/GAN_Ropa_" + str(i+1) + "_" + str(G_perdida_valor) + ".pt"
    scripted.save(archivo)
    plotearGenerados()
plotearPerdidas()

print("La perdida minima fue: {minPerdida} en la Epoca: {minEpoca}")
horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo GAN de Ropas en {tiempoSeg} seg")