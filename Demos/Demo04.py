import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Cargar el Modelo PreEntrenado AE")
modelo=torch.jit.load('preentrenados/AE_Ropa_76_0.006871494930237532.pt',map_location=device)
modelo.eval()

print("2. Crear el Transformador para los datos de prueba")
transformacion_data = T.Compose([T.ToTensor()])

print("3. Crear el DataSet de Pruebas")
X_test = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=transformacion_data)
print("X_test", X_test)

numEntradas = X_test[0][0].nelement()
print("numEntradas", numEntradas)

print("4. Crear el DataLoader de Pruebas")
batchSize = 32
loader_test = torch.utils.data.DataLoader(X_test, batch_size=batchSize, shuffle=True)

print("5. Mostrar las imagenes originales y las reconstruidas Pre-Entrenadas")
originales = []
indice = 0
for img, etiqueta in X_test:
    if(etiqueta==indice):
        originales.append(img)
        indice += 1
    if(indice==10):
        break

def plotearImagenes():
    reconstruidas = []
    for i in range(10):
        imgOriginal = originales[i].reshape((1,numEntradas))
        imgReconstruida, mu = modelo(imgOriginal.to(device))
        reconstruidas.append(imgReconstruida)
    imagenes = originales + reconstruidas

    plt.figure(figsize=(10,2),dpi=50)
    for i in range(20):
        ax = plt.subplot(2, 10, i + 1)
        img = imagenes[i].detach().cpu().numpy().reshape(28,28)
        plt.imshow(img, cmap="binary")
    plt.show()

plotearImagenes()