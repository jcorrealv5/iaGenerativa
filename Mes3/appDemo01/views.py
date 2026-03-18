from django.http import HttpResponse
from django.shortcuts import render
import torch, cv2
import numpy as np
from io import BytesIO
from PIL import Image

def ConsultaRopa(request):
    return render(request, "appDemo01/ConsultaRopa.html")

def GenerarRopa(request):
    rpta = None
    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','DCGAN', pretrained=True, useGPU=use_gpu)
    ruido, _ = model.buildNoiseData(1)
    with torch.no_grad():
        imagenesGeneradas = model.test(ruido)
    img = imagenesGeneradas[0]/2+0.5
    img = (img.clamp(0, 1) * 255).byte()
    img = img.permute(1,2,0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rpta = convertirNumPyToBytes(img)
    return HttpResponse(rpta)

def convertirNumPyToBytes(imagen):
    imagenPIL = Image.fromarray(imagen)
    imagenBuffer = BytesIO()
    imagenPIL.save(imagenBuffer, format="PNG")
    rpta = imagenBuffer.getvalue()
    return rpta