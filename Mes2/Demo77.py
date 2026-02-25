import torch
from modDifusion import DDPM, DummyEpsModel, Plotear
from datetime import datetime

print("Demo 77: Probando el Modelo de Difusion MNIST PreEntrenado")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Tipo salida: {device}")

horaInicio = datetime.now()

print("1. Crear el Modelo de Difusion")
modelo = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000).to(device)

print("2. Cargar los Pesos del Modelo PreEntrenado")
checkpoint = torch.load('/Users/jhon.correal/Documents/Python/Shifu/preentrenados/DM/DM00_008.pth', map_location=device)
modelo.load_state_dict(checkpoint)

print("3. Ejecutar el Modelo para Generar Imagenes")
modelo.eval()
with torch.no_grad():
    imagenesGeneradas = modelo.sample(8, (1, 28, 28), device)
    print(imagenesGeneradas.shape)

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print("4. Plotear las Imagenes Generadas")
Plotear.Imagenes(imagenesGeneradas, 2, 4, "Digitos MNIST con Modelos de Difusion")

print(f"5. Termino el Proceso de Generacion en: {tiempo} seg")