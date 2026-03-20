from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt
import torch, torchvision, cv2
import torchvision.transforms as T
import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from io import BytesIO
from PIL import Image
from modGAN import LoadData, Generador

def CambioRisa(request):
    return render(request, "appDemo23/CambioRisa.html")

@xframe_options_exempt
def CambiarRisa(request):
    rpta = None
    risa = request.POST.get("risa")
    print("risa: ", risa)
    fileOrigen = request.FILES["archivo"]
    fileOrigen.open()
    bytes = fileOrigen.read()
    fileOrigen.close()
    imagenRisaActual = convertirBytesToNumPy(bytes)
    batch_size = 1
    transforms = albumentations.Compose(
    [albumentations.Resize(width=256, height=256),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],max_pixel_value=255),
        ToTensorV2()],
    additional_targets={"image0": "image"})
    archivoSinRisa = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSonrisa/No/003853.jpg"
    archivoConRisa = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSonrisa/Si/003845.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generador(img_channels=3, num_residuals=9).to(device)
    if(risa=="0"):
        imagenRisaOpuesto = cv2.imread(archivoConRisa)        
        gen.load_state_dict(torch.load("C:/Data/Python/2026_01_IAG/Demos/preentrenados/GAN/Sonrisa/gen_conrisa_62.pth",map_location=device))
    else:
        imagenRisaOpuesto = cv2.imread(archivoSinRisa)
        gen.load_state_dict(torch.load("C:/Data/Python/2026_01_IAG/Demos/preentrenados/GAN/Sonrisa/gen_sinrisa_62.pth",map_location=device))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    imagenRisaOpuesto = cv2.resize(imagenRisaOpuesto, (256,256))
    imagenRisaActual = cv2.cvtColor(imagenRisaActual, cv2.COLOR_BGR2RGB)
    imagenRisaActual = cv2.resize(imagenRisaActual, (256,256))
    if(risa=="0"):
        augmentations = transforms(image=imagenRisaActual, image0=imagenRisaOpuesto)
    else:
        augmentations = transforms(image=imagenRisaOpuesto, image0=imagenRisaActual)
    imgConRisa = augmentations["image0"]
    imgSinRisa = augmentations["image"]
    dstRisa = [(imgSinRisa, imgConRisa)]
    loaderRisa=torch.utils.data.DataLoader(dstRisa,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    for noSonrie,siSonrie in loaderRisa:
        if(risa=="0"):
            fakeRisa=gen(noSonrie.to(device)).squeeze(0)
        else:
            fakeRisa=gen(siSonrie.to(device)).squeeze(0)
    imgRpta = fakeRisa/2+0.5
    imgRpta = (imgRpta.clamp(0, 1) * 255).byte()
    imgRpta = imgRpta.permute(1,2,0).cpu().numpy()
    imgRpta = cv2.cvtColor(imgRpta, cv2.COLOR_RGB2BGR)
    print("shape imgRpta: ", imgRpta.shape)
    imagen = convertirNumPyToBytes(imgRpta)
    return HttpResponse(imagen)

def convertirNumPyToBytes(imagen):
    imagenPIL = Image.fromarray(imagen)
    imagenBuffer = BytesIO()
    imagenPIL.save(imagenBuffer, format="PNG")
    rpta = imagenBuffer.getvalue()
    return rpta

def convertirBytesToNumPy(buffer):
    imagenPIL = Image.open(BytesIO(buffer))
    imagen = np.array(imagenPIL)
    return imagen