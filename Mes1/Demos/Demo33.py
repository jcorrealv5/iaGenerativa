alumnos_path = r"C:\Data\Python\2026_01_IAG\Demos\datasets\Alumnos"
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import multiprocessing as mp

def main():
    print("Demo 33: DC-GAN para Entrenar Caras de Alumnos")

    def plotearImagenes(imagenes, filas, cols):
        figura, ejes = plt.subplots(filas,cols)
        for i in range(filas):
            for j in range(cols):
                n = (i * cols) + j
                imgCara = np.transpose(imagenes[n].numpy(),(1,2,0))
                ejes[i, j].imshow(imgCara)
        plt.show()

    print("1. Crear el DataSet y DataLoader de Entrenamiento")
    transform = T.Compose([T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_data = ImageFolder(root=alumnos_path, transform=transform)
    batch_size = 64
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    imagenes, _ = next(iter(train_loader))
    plotearImagenes(imagenes, 4, 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
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
        nn.Flatten()).to(device)

    G=nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
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
        nn.Tanh()).to(device)

    print("2. Crear la Funcion de Perdida y los Optmiadores del D y G")
    loss_fn=nn.BCELoss()
    lr = 0.0002
    optimG = torch.optim.Adam(G.parameters(), lr = lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(D.parameters(), lr = lr, betas=(0.5, 0.999))

    def train_D_on_real(real_samples):
        real_samples=real_samples.to(device)
        preds=D(real_samples)
        labels=torch.ones((real_samples.shape[0],1)).to(device)
        loss_D=loss_fn(preds,labels)
        optimD.zero_grad()
        loss_D.backward()
        optimD.step()
        return loss_D

    def train_D_on_fake():
        noise=torch.randn(batch_size,100,1,1).to(device)
        generated_data=G(noise)
        preds=D(generated_data)
        fake_labels=torch.zeros((batch_size,1)).to(device)
        loss_D=loss_fn(preds,fake_labels)
        optimD.zero_grad()
        loss_D.backward()
        optimD.step()
        return loss_D

    def train_G():
        noise=torch.randn(batch_size,100,1,1).to(device)
        generated_data=G(noise)
        preds=D(generated_data)
        real_labels=torch.ones((batch_size,1)).to(device)
        loss_G=loss_fn(preds,real_labels)
        optimG.zero_grad()
        loss_G.backward()
        optimG.step()
        return loss_G

    def test_epoch():
        noise=torch.randn(32,100,1,1).to(device=device)
        fake_samples=G(noise).cpu().detach()
        for i in range(32):
            ax = plt.subplot(4, 8, i + 1)
            img=(fake_samples.cpu().detach()[i]/2+0.5).permute(1,2,0)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(hspace=-0.6)
        plt.show()

    print("3. Entrenar el Discriminador y el Generador")
    epocas = 1000
    nMuestras = len(train_loader)
    for i in range(epocas):
        gloss=0
        dloss=0
        for n, (real_samples,_) in enumerate(train_loader):
            print(f"item: {n+1} - bucles: {nMuestras} - epoca: {i+1}")
            loss_D=train_D_on_real(real_samples)
            dloss+=loss_D
            loss_D=train_D_on_fake()
            dloss+=loss_D
            loss_G=train_G()
            gloss+=loss_G
        gloss=gloss/n
        dloss=dloss/n
        G_perdida_valor = gloss.item()
        print(f"Epoca {i+1}, Perdida Discriminador: {dloss}, Perdida Generador: {gloss}")
        if(i % 10 == 0):
            scripted = torch.jit.script(G)
            archivo = "preentrenados/GAN/Alumnos/GAN_Alumnos_" + str(i+1) + "_" + str(G_perdida_valor) + ".pt"
            scripted.save(archivo)
    scripted = torch.jit.script(G)
    archivo = "preentrenados/GAN/Alumnos/GAN_Alumnos_" + str(i+1) + "_" + str(G_perdida_valor) + ".pt"
    scripted.save(archivo)
    test_epoch()

if __name__ == "__main__":
    mp.freeze_support()
    main()