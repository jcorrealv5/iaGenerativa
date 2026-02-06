import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
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
        self.label_emb = nn.Embedding(5, 64*64)
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )
    
    def forward(self, x, labels):
        c = self.label_emb(labels).view(labels.size(0), 1, 64, 64)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()

class Generador(nn.Module):
    def __init__(self):
        super().__init__()        
        self.label_emb = nn.Embedding(5, 50)        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(150, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100, 1, 1)
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, c], 1)
        x = x.view(x.size(0), 150, 1, 1)
        out = self.model(x)
        return out

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

    alumnos_path = r"C:\Data\Python\2026_01_IAG\Demos\datasets\Alumnos"
    print("Demo 39: Entrenamiento de una cGAN para Generar Caras de Alumnos")

    print("1. Crear el Transformador para los datos")
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    print("2. Crear el DataSet y DataLoader de Entrenamiento con Caras de Alumnos")
    X_train = ImageFolder(root=alumnos_path, transform=transform)
    print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
    batch_size = 32
    epocas = 1000
    loader_train = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, drop_last=True)

    print("4. Cargar un lote de imagenes y plotearlas")
    imagenes, etiquetas = next(iter(loader_train))
    plotearImagenes(imagenes, etiquetas, 4, 8)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device usado: ", device)
    if device == "cuda":
        print("Se esta usando Cuda: " + torch.cuda.get_device_name(0))
    else :
        print("Se esta usando CPU")

    print("5. Crear el Modelo para el generador y discriminador")
    generador = Generador().cuda()
    discriminador = Discriminador().cuda()

    print("6. Definir la Funcion de Perdida o Error y el Metodo de Optimizacion")
    criterio = nn.BCELoss()
    lr = 1e-4
    d_optimizer = torch.optim.Adam(discriminador.parameters(), lr=lr, betas=(0.0, 0.999),eps=1e-8)
    g_optimizer = torch.optim.Adam(generador.parameters(), lr=lr, betas=(0.0, 0.999),eps=1e-8)

    def entrenarGenerador(batch_size, discriminator, generator, g_optimizer, criterion):
        g_optimizer.zero_grad()
        ruido = torch.randn(batch_size, 100)
        z = Variable(ruido).cuda()
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 5, batch_size))).cuda()
        fake_images = generador(z, fake_labels)
        validity = discriminador(fake_images, fake_labels)
        g_loss = criterio(validity, Variable(torch.ones(batch_size)).cuda())
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()

    def entrenarDiscriminador(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
        d_optimizer.zero_grad()
        # train with real images
        real_validity = discriminador(real_images, labels)
        real_loss = criterio(real_validity, Variable(torch.ones(batch_size)).cuda())
        # train with fake images
        z = Variable(torch.randn(batch_size, 100)).cuda()
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 5, batch_size))).cuda()
        fake_images = generador(z, fake_labels)
        fake_validity = discriminador(fake_images, fake_labels)
        fake_loss = criterio(fake_validity, Variable(torch.zeros(batch_size)).cuda())    
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        return d_loss.item()

    print("7. Entrenar el Modelo cGAN a 1000 Epocas")
    num_epochs = 2000
    nMuestras = len(loader_train)
    n_critic = 5
    generador.train()
    discriminador.train()
    for epoca in range(num_epochs):
        for i, (images, labels) in enumerate(loader_train):
            print(f"item: {i+1} - bucles: {nMuestras} - epoca: {epoca+1}")
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()              
            for j in range(n_critic):
                d_loss = entrenarDiscriminador(len(real_images), discriminador, generador, d_optimizer, criterio, real_images, labels)
            g_loss = entrenarGenerador(batch_size, discriminador, generador, g_optimizer, criterio)
        print(f"Epoca {epoca+1}, Perdida Discriminador: {d_loss}, Perdida Generador: {g_loss}")
        if(g_loss<5):
            scripted = torch.jit.script(generador)
            archivo = "preentrenados/GAN/Alumnos/cGAN_Alumnos_" + str(epoca+1) + "_" + str(g_loss) + ".pt"
            scripted.save(archivo)
    scripted = torch.jit.script(generador)
    archivo = "preentrenados/GAN/Alumnos/cGAN_Alumnos_" + str(epoca+1) + "_" + str(g_loss) + ".pt"
    scripted.save(archivo)

    horaFin = datetime.now()
    tiempoSeg = (horaFin - horaInicio).total_seconds()
    print(f"Se creo y entreno el Modelo cGAN de Caras de Alumnos en {tiempoSeg} seg")

if __name__ == "__main__":
    mp.freeze_support()
    main()