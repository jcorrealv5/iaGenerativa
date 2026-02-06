import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# ------------------ MODELOS ------------------

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
        return self.model(x).squeeze()


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

# ------------------ MAIN ------------------

def main():
    horaInicio = datetime.now()

    def plotearImagenes(imagenes, etiquetas, filas, cols):
        fig, ejes = plt.subplots(filas, cols)
        for i in range(filas):
            for j in range(cols):
                n = (i * cols) + j
                ejes[i, j].imshow(imagenes[n].squeeze(), cmap="gray")
                ejes[i, j].set_title(etiquetas[n].item())
                ejes[i, j].axis("off")
        plt.show()

    print("Demo 36: Entrenamiento de una cGAN para Generar Digitos de MNIST")

    transformacion_data = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    dataset = MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transformacion_data
    )

    batch_size = 64
    loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True
    )

    imagenes, etiquetas = next(iter(loader_train))
    plotearImagenes(imagenes, etiquetas, 4, 8)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device usado:", device)

    generador = Generador().to(device)
    discriminador = Discriminador().to(device)

    criterio = nn.BCELoss()
    lr = 1e-4
    d_optimizer = torch.optim.Adam(discriminador.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generador.parameters(), lr=lr)

    def entrenarGenerador(batch_size):
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, 100, device=device)
        fake_labels = torch.randint(0, 10, (batch_size,), device=device)
        fake_images = generador(z, fake_labels)
        validity = discriminador(fake_images, fake_labels)
        g_loss = criterio(validity, torch.ones(batch_size, device=device))
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()

    def entrenarDiscriminador(real_images, labels):
        d_optimizer.zero_grad()

        real_validity = discriminador(real_images, labels)
        real_loss = criterio(real_validity, torch.ones(len(real_images), device=device))

        z = torch.randn(len(real_images), 100, device=device)
        fake_labels = torch.randint(0, 10, (len(real_images),), device=device)
        fake_images = generador(z, fake_labels)
        fake_validity = discriminador(fake_images, fake_labels)
        fake_loss = criterio(fake_validity, torch.zeros(len(real_images), device=device))

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        return d_loss.item()

    print("Entrenando cGAN...")
    num_epochs = 10

    for epoca in range(num_epochs):
        for images, labels in loader_train:
            images = images.to(device)
            labels = labels.to(device)

            d_loss = entrenarDiscriminador(images, labels)
            g_loss = entrenarGenerador(batch_size)

        print(f"Epoca {epoca+1} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

        scripted = torch.jit.script(generador)
        scripted.save(f"preentrenados/GAN/MNIST/cGAN_Digitos_{epoca+1}.pt")

    tiempo = (datetime.now() - horaInicio).total_seconds()
    print(f"Entrenamiento finalizado en {tiempo:.2f} segundos")


if __name__ == "__main__":
    main()