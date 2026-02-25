import torch.nn as nn
import torch, os
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from torch.utils.data import Dataset
import albumentations 
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from modGAN import LoadData, Discriminador, Generador

horaInicio = datetime.now()
print("Demo 66: Entrenamiento de un Modelo Cycle-GAN de CelebA para Bigotes y Barba")

def iniciarPesos(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1 or name.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif name.find('Norm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0) 
    
def test(epoca,i,A,B,fake_A,fake_B):
    carpeta = "C:/Data/Python/2026_01_IAG/Demos/preentrenados/GAN/Bigote/Epoca" + str(epoca)
    if(not os.path.isdir(carpeta)):
        os.makedirs(carpeta)
    save_image(A*0.5+0.5, os.path.join(carpeta, f"A{i}.png"))
    save_image(B*0.5+0.5, os.path.join(carpeta, f"B{i}.png"))
    save_image(fake_A*0.5+0.5,os.path.join(carpeta, f"fakeA{i}.png"))
    save_image(fake_B*0.5+0.5,os.path.join(carpeta, f"fakeB{i}.png"))

def train_epoch(epoca,disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,device):
    total_batches = len(loader)
    for i, (A,B) in enumerate(loader):
        A=A.to(device)
        B=B.to(device)
        # Train Discriminators A and B
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss
            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss
            # Average loss of the two discriminators
            D_loss = (D_A_loss + D_B_loss) / 2
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # Train the two generators 
        with torch.cuda.amp.autocast():
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
            # NEW in Cycle GANs: cycle loss
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)
            # Total generator loss
            G_loss=(loss_G_A+loss_G_B+cycle_A_loss*10+cycle_B_loss*10)
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()        
        # Print progress
        print(f"Epoca: {epoca} - Item: {i+1}/{total_batches} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")
        if i % 100 == 0:
            test(epoca,i,A,B,fake_A,fake_B)
batch_size=8

print("1. Cargar el DataSet y DataLoader de CelebA")
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
dataset = LoadData(root_A=["datasets/Bigotes/No/"],root_B=["datasets/Bigotes/Si/"],transform=transforms)
loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

print("2. Crear los Discriminadores e iniciar sus pesos")
device = "cuda" if torch.cuda.is_available() else "cpu"
disc_A = Discriminador().to(device)
disc_B = Discriminador().to(device)
iniciarPesos(disc_A)
iniciarPesos(disc_B)

print("3. Crear los Generadores e iniciar sus pesos")
gen_A = Generador(img_channels=3, num_residuals=9).to(device)
gen_B = Generador(img_channels=3, num_residuals=9).to(device)
iniciarPesos(gen_A)
iniciarPesos(gen_B)

print("4. Definir Funciones de Perdida y Optimizacion")
l1 = nn.L1Loss()
mse = nn.MSELoss()
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
lr = 0.00001
opt_disc = torch.optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()),lr=lr,betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(gen_A.parameters()) + list(gen_B.parameters()),lr=lr,betas=(0.5, 0.999))

print("5. Entrenar los Modelos Discriminador y Generador")
for epoca in range(100):
    train_epoch(epoca+1,disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, device)
    torch.save(gen_A.state_dict(), "preentrenados/GAN/Bigote/gen_sinbigote_" + str(epoca+1) + ".pth")
    torch.save(gen_B.state_dict(), "preentrenados/GAN/Bigote/gen_conbigote_" + str(epoca+1) + ".pth")
torch.save(gen_A.state_dict(), "preentrenados/GAN/Bigote/gen_sinbigote.pth")
torch.save(gen_B.state_dict(), "preentrenados/GAN/Bigote/gen_conbigote.pth")

horaFin = datetime.now()
tiempoSeg = (horaFin - horaInicio).total_seconds()
print(f"Se creo y entreno el Modelo Cycle GAN de Bigotes y Barbas en {tiempoSeg} seg")