import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime

horaInicio = datetime.now()

print("1. Crear el Transformador para los datos")
transformacion_data = T.Compose([T.ToTensor()])

print("2. Crear los DataSets de Entrenamiento y Pruebas")
X_train = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=transformacion_data)
print("X_train", X_train)
X_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
print("X_test", X_test)

print("3. Crear los DataLoaders de Entrenamiento y Pruebas")
batchSize = 100
loader_train = torch.utils.data.DataLoader(X_train, batch_size=batchSize, shuffle=True)
loader_test = torch.utils.data.DataLoader(X_test, batch_size=batchSize, shuffle=True)

def plotearDigitos(imagenes):
    plt.figure(figsize=(10,10),dpi=50)
    for i in range(100):
        ax = plt.subplot(10, 10, i + 1)
        img = imagenes[i].detach().cpu().numpy().reshape(28,28)
        plt.imshow(img, cmap="binary")
    plt.show()

print("4. Plotear los 100 Primeros Digitos")
num_samples = 100
imagenes, etiquetas = next(iter(loader_train))
print(imagenes.shape)
plotearDigitos(imagenes)

print("5. Crear un Modelo VAE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

modelo = VAE().to(device)
optimizer = Adam(modelo.parameters(), lr=1e-3)

epocas = 100
nMuestras = len(loader_train)
def funcion_perdida(x, x_hat, mean, log_var):
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD

print("6. Entrenar el Modelo VAE de Digitos")
modelo.train()
perdida_promedio = 0
for epoca in range(epocas):
    perdida_total = 0
    for i, (x, _) in enumerate(loader_train):
        print(f"Item: {i+1}/{nMuestras}/{epoca+1}")
        x = x.view(batchSize, 784).to(device)
        optimizer.zero_grad()
        x_hat, mean, log_var = modelo(x)
        perdida = funcion_perdida(x, x_hat, mean, log_var)
        perdida_total += perdida.item()
        perdida.backward()
        optimizer.step()
        perdida_promedio = perdida_total/((i+1)*batchSize)
    print("\tEpoca", epoca + 1, "\tPerdida Promedio: ", perdida_promedio)
    scripted = torch.jit.script(modelo)
    scripted.save("preentrenados/VAE/MNIST/VAE_Digitos_" + str(epoca+1) + "_" + str(perdida_promedio) + ".pt")

horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo AE de Ropa en {tiempoSeg} seg")