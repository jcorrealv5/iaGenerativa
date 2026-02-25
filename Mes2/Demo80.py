import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset
import multiprocessing as mp

def main():
    print("Demo 80: Usando denoising_diffusion_pytorch para entrenar data personalizada")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Tipo salida: {device}")

    print("1. Creando un Modelo con la Arquitectura UNet")
    modelo = Unet(dim = 64, dim_mults = (1, 2, 4)).to(device)

    print("2. Creando un Objeto de Difusion Gaussiano con el Modelo")
    diffusion = GaussianDiffusion(modelo, image_size = 128, timesteps = 1000).to(device)

    print("3. Creando un objeto Trainer para el Modelo de Difusion")
    trainer = Trainer(
        diffusion,
        '/Users/jhon.correal/Documents/Python/Shifu/datasets/Alumnos/',
        train_batch_size = 8,
        train_lr = 1e-5,
        train_num_steps = 100,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        amp = True,
        save_and_sample_every = 10,
        results_folder = '/Users/jhon.correal/Documents/Python/Shifu/preentrenados/DM/Demo80',
        calculate_fid = False   # 👈 ESTE ES EL CAMBIO
    )

    print("4. Entrenando el Modelo")
    trainer.train()

    print("5. Guardando el modelo entrenado")
    torch.save(diffusion, "/Users/jhon.correal/Documents/Python/Shifu/preentrenados/DM/Demo80.pth")

    print("6. Fin del Programa")

if __name__ == "__main__":
    mp.freeze_support()
    main()