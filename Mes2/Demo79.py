import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image

print("Demo 79: Usando denoising_diffusion_pytorch para generar ruido desde un Modelo PreEntrenado")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Tipo salida: {device}")

print("1. Creando un Modelo con la Arquitectura UNet")
modelo = Unet(dim = 64, dim_mults = (1, 2, 4, 8)).to(device)

print("2. Creando un Objeto de Difusion Gaussiano con el Modelo")
diffusion = GaussianDiffusion(modelo, image_size = 128, timesteps = 1000).to(device)

print("3. Cargando el Modelo de Difusion Entrenado")
checkpoint = torch.load('/Users/jhon.correal/Documents/Python/Shifu/preentrenados/DM/Demo78.pth', map_location=device)
diffusion.load_state_dict(checkpoint)

print("4. Configurando al Modo de Test")
diffusion.eval()

print("5. Generando 4 imagenes de muestra")
with torch.no_grad():
    sampled_images = diffusion.sample(batch_size = 4)
    print("Shape sampled_images: ", sampled_images.shape)

print("6. Guardando las imagenes de muestra")
save_image(sampled_images, "Demo79_Ruido.png", nrow=2, normalize=True)

print("7. Fin del Programa")