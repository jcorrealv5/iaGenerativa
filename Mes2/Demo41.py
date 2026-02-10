import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

print("Demo 41: Generando Caras de Alumnos en Disco usando cGAN")

# --------------------------------------------------
# 1. Detectar automáticamente si hay GPU o no
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# --------------------------------------------------
# 2. Cargar modelo preentrenado en el dispositivo correcto
# --------------------------------------------------
archivo_modelo = "preentrenados/GAN/Alumnos/cGAN_Alumnos_1594_0.08066758513450623.pt"
generador = torch.jit.load(archivo_modelo, map_location=device)
generador.eval()  # Modo inferencia

# --------------------------------------------------
# 3. Inputs del usuario
# --------------------------------------------------
idAlumno = int(input("Ingresa el id del Alumno de 0 a 4 a generar: "))
batch_size = int(input("Cuantos archivos deseas generar con caras: "))

# --------------------------------------------------
# 4. Crear ruido latente y labels (SIN .cuda())
# --------------------------------------------------
z = torch.randn(batch_size, 100, device=device)

labels = torch.LongTensor(
    [idAlumno for _ in range(batch_size)]
).to(device)

# --------------------------------------------------
# 5. Generar imágenes
# --------------------------------------------------
with torch.no_grad():  # No necesitamos gradientes
    imagenesGeneradas = generador(z, labels).cpu()

# --------------------------------------------------
# 6. Crear carpeta de salida
# --------------------------------------------------
carpeta = str(idAlumno)
if not os.path.isdir(carpeta):
    os.makedirs(carpeta)

# --------------------------------------------------
# 7. Post-procesamiento y guardado con OpenCV
# --------------------------------------------------
for i in range(batch_size):
    print(f"Creando imagen: {i + 1}")

    # Normalizar de [-1,1] → [0,1]
    img = imagenesGeneradas[i] / 2 + 0.5

    # Convertir a uint8 [0,255]
    img = (img.clamp(0, 1) * 255).byte()

    # CHW → HWC
    img = img.permute(1, 2, 0).numpy()

    # RGB → BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Redimensionar
    img = cv2.resize(img, (200, 200))

    # Guardar imagen
    archivo_salida = os.path.join(carpeta, f"{i + 1}.png")
    cv2.imwrite(archivo_salida, img)

print(f"Se crearon: {batch_size} archivos con caras del alumno: {idAlumno}")