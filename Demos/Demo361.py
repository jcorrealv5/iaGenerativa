import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as T
from torch import autograd
from torch.autograd import Variable
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

class Discriminador(nn.Module):
    def __init__(self):
        super().__init__()        
        self.label_emb = nn.Embedding(10, 10)        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()

class Generador(nn.Module):
    def __init__(self):
        super().__init__()        
        self.label_emb = nn.Embedding(10, 10)        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

def main():
    horaInicio = datetime.now()

    def plotearImagenes(imagenes, etiquetas, filas, cols):
        figura, ejes = plt.subplots(filas,cols)
        for i in range(filas):
            for j in range(cols):
                n = (i * cols) + j
                valor = etiquetas[n].item()
                imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
                ejes[i, j].imshow(imgCara, cmap="gray")
                ejes[i, j].set_title(valor)
        plt.show()

    print("Demo 36: Entrenamiento de una cGAN para Generar Digitos de MNIST")

    print("1. Crear el Transformador para los datos")
    transformacion_data = T.Compose([T.ToTensor(), T.Normalize([0.5],[0.5])])

    print("2. Crear el DataSet y DataLoader de Entrenamiento con MNIST")
    X_train = torchvision.datasets.MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transformacion_data
    )

    print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
    batch_size = 64
    epocas = 10

    loader_train = torch.utils.data.DataLoader(
        X_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,          # CAMBIO: evitar warning y bloqueos en Mac
        pin_memory=False,       # CAMBIO: pin_memory no aplica en CPU/MPS
        drop_last=True
    )

    print("3. Cargar un lote de imagenes y plotearlas")
    imagenes, etiquetas = next(iter(loader_train))
    plotearImagenes(imagenes, etiquetas, 4, 8)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )  # CAMBIO: device robusto (CPU / MPS / CUDA)

    print("Device usado: ", device)

    print("4. Crear el Modelo para el generador y discriminador")
    generador = Generador().to(device)       # CAMBIO: reemplazo de .cuda()
    discriminador = Discriminador().to(device)  # CAMBIO: reemplazo de .cuda()

    print("5. Definir la Funcion de Perdida o Error y el Metodo de Optimizacion")
    criterio = nn.BCELoss()
    lr = 1e-4
    d_optimizer = torch.optim.Adam(discriminador.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generador.parameters(), lr=lr)

    def entrenarGenerador(batch_size, discriminator, generator, g_optimizer, criterion):
        g_optimizer.zero_grad()
        ruido = torch.randn(batch_size, 100, device=device)  # CAMBIO: tensor en device
        z = Variable(ruido)
        fake_labels = Variable(
            torch.LongTensor(np.random.randint(0, 10, batch_size)).to(device)
        )  # CAMBIO
        fake_images = generador(z, fake_labels)
        validity = discriminador(fake_images, fake_labels)
        g_loss = criterio(validity, torch.ones(batch_size, device=device))  # CAMBIO
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()

    def entrenarDiscriminador(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
        d_optimizer.zero_grad()

        real_validity = discriminador(real_images, labels)
        real_loss = criterio(
            real_validity,
            torch.ones(batch_size, device=device)  # CAMBIO
        )

        z = torch.randn(batch_size, 100, device=device)  # CAMBIO
        fake_labels = torch.LongTensor(
            np.random.randint(0, 10, batch_size)
        ).to(device)  # CAMBIO
        fake_images = generador(z, fake_labels)
        fake_validity = discriminador(fake_images, fake_labels)
        fake_loss = criterio(
            fake_validity,
            torch.zeros(batch_size, device=device)  # CAMBIO
        )

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        return d_loss.item()

    print("6. Entrenar el Modelo cGAN")
    num_epochs = 100
    n_critic = 5
    nMuestras = len(loader_train)

    for epoca in range(num_epochs):
        for i, (images, labels) in enumerate(loader_train):
            print(f"item: {i+1} - bucles: {nMuestras} - epoca: {epoca+1}")

            real_images = images.to(device)   # CAMBIO: reemplazo .cuda()
            labels = labels.to(device)         # CAMBIO

            generador.train()
            d_loss = 0
            for _ in range(n_critic):
                d_loss = entrenarDiscriminador(
                    len(real_images),
                    discriminador,
                    generador,
                    d_optimizer,
                    criterio,
                    real_images,
                    labels
                )

            g_loss = entrenarGenerador(
                batch_size,
                discriminador,
                generador,
                g_optimizer,
                criterio
            )

        print(f"Epoca {epoca+1}, Perdida Discriminador: {d_loss}, Perdida Generador: {g_loss}")

        scripted = torch.jit.script(generador)
        archivo = f"preentrenados/GAN/MNIST/cGAN_Digitos_{epoca+1}_{g_loss}.pt"
        scripted.save(archivo)

    horaFin = datetime.now()
    tiempoSeg = (horaFin - horaInicio).total_seconds()
    print(f"Se creo y entreno el Modelo cGAN de Digitos en {tiempoSeg} seg")

if __name__ == "__main__":
    main()