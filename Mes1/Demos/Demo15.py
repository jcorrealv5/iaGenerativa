import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Cargar el Modelo PreEntrenado AE")
modelo=torch.jit.load('preentrenados/VAE/Caras/VAE_Caras_998_79.27378463745117.pt',map_location=device)
modelo.eval()

print("2. Crear el Transformador para los datos de prueba")
transform = T.Compose([T.ToTensor(), T.Resize(100)])

print("3. Crear los DataSets y DataLoader de Pruebas")
dataset_test = torchvision.datasets.ImageFolder(root="datasets/Voluntarios", transform=transform)
batch_size=60
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=True)

print("4. Mostrar las imagenes originales y las reconstruidas Pre-Entrenadas")
def generarCara(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = modelo.decode(z_sample)
    caraGenerada = x_decoded.detach().cpu().reshape(3, 100, 100)
    plt.imshow(caraGenerada, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_latent_space(model, scale=2.5, n=8, face_size=100, figsize=200):
    grid_x = np.linspace(-scale, scale, n)    
    grid_y = np.linspace(-scale, scale, n)[::-1]
    #print("shape grid_x: ", grid_x)
    #print("shape grid_y: ", grid_y)
    figura, ejes = plt.subplots(n,n)
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            print(f"Media: {xi} y Varianza: {yi}")
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            caraTensor = x_decoded[0].detach().cpu().reshape(3,face_size, face_size)
            caraArray = np.transpose(caraTensor.numpy(),(1,2,0))
            ejes[i, j].imshow(caraArray)
    plt.show()
plot_latent_space(modelo)
#Plotear el 1 que tiene media 0 y var 1
