from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from modDifusion import DDPM, DummyEpsModel

def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    print("Demo 76: Creando un Modelo de Difusion desde Cero en PyTorch para Generar Digitos MNIST")
    print("1. Crear un modelo de tipo DDPM")
    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)
    print("2. Crear un objeto para Transformar Datos")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    print("2. Cargar el DataSet de Digitos MNIST")
    dataset = MNIST(root="datasets/MNIST",train=True,download=True,transform=tf)
    print("3. Crear el DataLoader de Digitos MNIST")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    print("4. Definir la Funcion de Optimizacion")
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    print("5. Entrenar el Modelo de Difusion")
    for i in range(n_epoch):
        ddpm.train()
        batch_count = 0
        total_batches = len(dataloader)
        loss_ema = None
        for x, _ in dataloader:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            batch_count += 1
            print(f"Epoca: {i+1}/{n_epoch} - Batch {batch_count}/{total_batches} - Loss: {loss_ema:.4f}")
            optim.step()
        print(f"Epoca {i+1}/{n_epoch} completada - Loss Final: {loss_ema:.4f}")
        ddpm.eval()
        print(f"Grabar la Imagen de la Epoca: {i+1}")
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"preentrenados/DM/Demo76/ddpm_sample_{i+1}.png")
            torch.save(ddpm.state_dict(), f"preentrenados/DM/Demo76_{i+1}.pth")
    print("6. Fin del Proceso")
if __name__ == "__main__":
    train_mnist()