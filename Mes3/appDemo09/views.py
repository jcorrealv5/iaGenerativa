from django.http import HttpResponse
from django.shortcuts import render
import torch, torchvision, cv2
import torchvision.transforms as T
import numpy as np
from io import BytesIO
from PIL import Image

def GenerarCaraAlumno(request):
    return render(request, "appDemo09/GenerarCaraAlumno.html")

def GenerarCaras(request):
    rpta = None
    caraInicio = int(request.GET.get("caraInicio"))
    caraFin = int(request.GET.get("caraFin"))
    nMuestras = int(request.GET.get("nMuestras"))
    print("caraInicio:", caraInicio)
    print("caraFin:", caraFin)
    print("nMuestras:", nMuestras)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo=torch.jit.load('C:/Data/Python/2026_01_IAG/Demos/preentrenados/VAE/Caras/VAE_Caras_Final_79.35767582484654.pt',map_location=device)
    modelo.eval()
    transform = T.Compose([T.ToTensor(), T.Resize(100)])
    dataset_test = torchvision.datasets.ImageFolder(root="C:/Data/Python/2026_01_IAG/Demos/datasets/Voluntarios", transform=transform)
    batch_size=60
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=True)
    X_test, y_test = next(iter(loader_test))
    x1 = X_test[y_test == caraInicio][1].to(device)
    x2 = X_test[y_test == caraFin][1].to(device)
    x1 = x1.view(1, 3*100*100).to(device)
    mean1, logvar1 = modelo.encode(x1)
    z1 = modelo.reparameterization(mean1, logvar1)
    x2 = x2.view(1, 3*100*100).to(device)
    mean2, logvar2 = modelo.encode(x2)
    z2 = modelo.reparameterization(mean2, logvar2)
    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, nMuestras)])
    listaInterpolada = modelo.decode(z)
    arrayInterpolado = listaInterpolada.to('cpu').detach().numpy()
    w = 100
    imagenes = []
    texto = ""
    for i, x_hat in enumerate(arrayInterpolado):
        img_array = x_hat.reshape(3, w, w)
        img_array = np.transpose(img_array,(1,2,0)) 
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_final = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        imgBytes = convertirNumPyToBytes(img_final)
        imagenes.append(imgBytes)
        texto += str(len(imgBytes))
        if(i<nMuestras-1):
            texto += "|"
    nTexto = len(texto)
    bytesRpta = []
    byte1 = int(nTexto / 255)
    byte2 = int(nTexto % 255)
    bytesTexto = texto.encode(encoding="utf-8")
    bytesRpta.append(byte1)
    bytesRpta.append(byte2)
    bytesRpta.extend(bytesTexto)
    for i in range(len(imagenes)):
        bytesRpta.extend(imagenes[i])
    rpta = bytes(bytesRpta)
    return HttpResponse(rpta)

def convertirNumPyToBytes(imagen):
    imagenPIL = Image.fromarray(imagen)
    imagenBuffer = BytesIO()
    imagenPIL.save(imagenBuffer, format="PNG")
    rpta = imagenBuffer.getvalue()
    return rpta