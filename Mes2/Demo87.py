import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from modDiffusers import ClassConditionedUnet
from datetime import datetime
from datasets import load_dataset

ds = load_dataset("mnist")

horaInicio = datetime.now()
print("Demo 87: Diffusers para Modelos de Difusion Condicionales con MNIST")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Tipo de dispositivo: ", device)

print("1. Crear el DataSet y DataLoader para MNIST")
dataset = torchvision.datasets.MNIST(root="/Users/jhon.correal/Documents/Python/Shifu/datasets", train=True, download=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("2. Cargar el primer lote de 8 y plotear")
x, y = next(iter(dataloader))
print("Shape x: ", x.shape)
print("Shape y: ", y.shape)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.show()

print("3. Crear el Modelo de Difusion")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
n_epocas = 100
modelo = ClassConditionedUnet().to(device)

print("4. Definir las Funciones de Perdida y Optimizacion")
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)

print("5. Entrenar el Modelo de Difusion")
losses = []
for epoca in range(n_epocas):
    nMuestras = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        print(f"Item: {i}/{nMuestras} - Epoca: {epoca+1}/{n_epocas}")
        x = x.to(device) * 2 - 1
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        # Get the model prediction
        pred = modelo(noisy_x, timesteps, y)
        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise
        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Store the loss for later
        losses.append(loss.item())
    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Epoca: {epoca + 1} - Perdida: {avg_loss:05f}')
    torch.save(modelo.state_dict(), f"preentrenados/DM/Demo87/Epoca{epoca+1}_{avg_loss:05f}.pth")
 
horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()
print(f"Tiempo Total de Entrenamiento Modelo Difusion Condicional: {tiempo} seg")