import torch
import torchvision
import torchvision.transforms as T
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Cargar el Modelo PreEntrenado AE")
modelo=torch.jit.load('preentrenados/AE_Caras_81_0.0014598657144233584.pt',map_location=device)
modelo.eval()

print("2. Crear el Transformador para los datos de prueba")
transform = T.Compose([T.ToTensor()])

print("2. Crear los DataSets y DataLoader de Pruebas")
X_test = torchvision.datasets.ImageFolder(root="datasets/Voluntarios", transform=transform)
batch_size=16
loader_test = torch.utils.data.DataLoader(X_test, batch_size=batch_size,shuffle=True)

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
    if(indice==5):
        break

def plotearImagenes():
    reconstruidas = []
    for i in range(5):
        imgOriginal = originales[i].reshape((1,30000))
        imgReconstruida, mu = modelo(imgOriginal.to(device))
        img_tensor = imgReconstruida.reshape((3, 100, 100))
        img_final = np.transpose(img_tensor.detach().cpu().numpy(), (1, 2, 0)) 
        reconstruidas.append(img_final)

    figura, ejes = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            if(i==0):
                ejes[i,j].imshow(np.transpose(originales[j],(1,2,0)), cmap="gray")
            else:
                ejes[i, j].imshow(reconstruidas[j])
    plt.show()

plotearImagenes()