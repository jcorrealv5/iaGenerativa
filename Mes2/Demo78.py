import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

print("Demo 78: Usando denoising_diffusion_pytorch para entrenar y guardar un Modelo")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Tipo salida: {device}")

print("1. Creando un Modelo con la Arquitectura UNet")
modelo = Unet(dim = 64, dim_mults = (1, 2, 4, 8)).to(device)

print("2. Creando un Objeto de Difusion Gaussiano con el Modelo")
diffusion = GaussianDiffusion(modelo, image_size = 128, timesteps = 1000).to(device)

print("3. Creando 8 Vectores de Ruido para Imagenes de 3x128x128")
training_images = torch.randn(8, 3, 128, 128).to(device)

print("4. Entrenando las imagenes con el Modelo de Difusion")
loss = diffusion(training_images)
loss.backward()

print("5. Guardando el modelo entrenado")
torch.save(diffusion.state_dict(), "/Users/jhon.correal/Documents/Python/Shifu/preentrenados/DM/Demo78.pth")

print("6. Fin del Programa")