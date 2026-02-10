import torch.nn as nn
import torch
import torchvision.transforms as T
import torchvision
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def main():
    horaInicio = datetime.now()

    print("Demo 42: Entrenamiento de una cGAN usando WGAN + WGAN-GP")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("1. Crear la Red del Discriminador o Critico")
    class Critico(nn.Module):
        def __init__(self, img_channels, features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(img_channels, features, 
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                self.block(features, features * 2, 4, 2, 1),
                self.block(features * 2, features * 4, 4, 2, 1),
                self.block(features * 4, features * 8, 4, 2, 1),
                self.block(features * 8, features * 16, 4, 2, 1),  
                self.block(features * 16, features * 32, 4, 2, 1),            
                nn.Conv2d(features * 32, 1, kernel_size=4,
                          stride=2, padding=0))
        def block(self, in_channels, out_channels, 
                  kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels,out_channels,
                    kernel_size,stride,padding,bias=False,),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2))
        def forward(self, x):
            return self.net(x)

    print("2. Crear la Red del Generador")
    class Generador(nn.Module):
        def __init__(self, noise_channels, img_channels, features):
            super(Generador, self).__init__()
            self.net = nn.Sequential(
                self.block(noise_channels, features *64, 4, 1, 0),
                self.block(features * 64, features * 32, 4, 2, 1),
                self.block(features * 32, features * 16, 4, 2, 1),
                self.block(features * 16, features * 8, 4, 2, 1),
                self.block(features * 8, features * 4, 4, 2, 1),            
                self.block(features * 4, features * 2, 4, 2, 1),            
                nn.ConvTranspose2d(
                    features * 2, img_channels, kernel_size=4,
                    stride=2, padding=1),
                nn.Tanh())
        def block(self, in_channels, out_channels, 
                  kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,
                    kernel_size,stride,padding,bias=False,),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),)
        def forward(self, x):
            return self.net(x)
    
    print("3. Crear la Funcion de Inciar Pesos")
    def iniciarPesos(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    print("4. Configurando Parametros e inicializando Pesos de ambas redes")
    z_dim=100
    img_channels=3
    features=16
    generador=Generador(z_dim+2,img_channels,features).to(device)
    critico=Critico(img_channels+2,features).to(device)
    iniciarPesos(generador)
    iniciarPesos(critico)

    print("5. Definir la Funcion de Optmizacion de ambas redes")
    lr = 0.0001
    opt_generador = torch.optim.Adam(generador.parameters(), lr = lr, betas=(0.0, 0.9))
    opt_critico = torch.optim.Adam(critico.parameters(), lr = lr, betas=(0.0, 0.9))

    print("6. Definir la Funcion de Penalizacion de Gradiente para el Critico")
    def GP(critico, real, fake):
        B, C, H, W = real.shape
        alpha=torch.rand((B,1,1,1)).repeat(1,C,H,W).to(device)
        interpolated_images = real*alpha+fake*(1-alpha)
        critic_scores = critico(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=critic_scores,
            grad_outputs=torch.ones_like(critic_scores),
            create_graph=True,
            retain_graph=True)[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gp = torch.mean((gradient_norm - 1) ** 2)
        return gp

    print("7. Cargar el DataSet de Glasses")
    batch_size=16
    imgsz=256
    transform=T.Compose([
        T.Resize((imgsz,imgsz)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])      
    data_set=torchvision.datasets.ImageFolder(root=r"datasets/Glasses", transform=transform) 

    print("8. Crea una Lista con las Imagenes, Etiquetas, Codificaciones Etiquetas y Juntas")
    newdataset=[]
    for i,(img,label) in enumerate(data_set):
        print(f"Procesando imagen {i} de {len(data_set)}")
        onehot=torch.zeros((2))
        onehot[label]=1
        channels=torch.zeros((2,imgsz,imgsz))
        if label==0:
            channels[0,:,:]=1
        else:
            channels[1,:,:]=1    
        img_and_label=torch.cat([img,channels],dim=0)    
        newdataset.append((img,label,onehot,img_and_label))

    print("9. Crear el DataLoader a partir de la lista anterior")
    data_loader=torch.utils.data.DataLoader(newdataset,batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, drop_last=True)

    print("10. Crear la Funcion para Plotear Imagenes x cada Epoca")
    def plotearEpoca(epoca):
        # test images with glasses
        noise = torch.randn(32, z_dim, 1, 1)
        labels = torch.zeros(32, 2, 1, 1)
        # use label [1,0] so G knows what to generate
        labels[:,0,:,:]=1
        noise_and_labels=torch.cat([noise,labels],dim=1).to(device)
        fake=generador(noise_and_labels).cpu().detach()
        fig=plt.figure(figsize=(20,10),dpi=100)
        for i in range(32):
            ax = plt.subplot(4, 8, i + 1)
            img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(hspace=-0.6)
        #plt.savefig(f"files/glasses/G{epoch}.png")
        plt.show() 
        # test images without glasses
        noise = torch.randn(32, z_dim, 1, 1)
        labels = torch.zeros(32, 2, 1, 1)
        # use label [0,1] so G knows what to generate
        labels[:,1,:,:]=1
        noise_and_labels=torch.cat([noise,labels],dim=1).to(device)
        fake=generador(noise_and_labels).cpu().detach()
        fig=plt.figure(figsize=(20,10),dpi=100)
        for i in range(32):
            ax = plt.subplot(4, 8, i + 1)
            img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(hspace=-0.6)
        #plt.savefig(f"files/glasses/NoG{epoch}.png")
        plt.show()

    print("11. Crear la Funcion para Entrenar cada Epoca")
    def entrenarEpoca(onehots,img_and_labels,epoca):
        real = img_and_labels.to(device)
        B = real.shape[0]
        # Entrenar el Critico 5 veces
        for _ in range(5):
            noise = torch.randn(B, z_dim, 1, 1)
            onehots=onehots.reshape(B,2,1,1)
            noise_and_labels=torch.cat([noise,onehots],dim=1).to(device)
            fake_img = generador(noise_and_labels).to(device)
            fakelabels=img_and_labels[:,3:,:,:].to(device)
            fake=torch.cat([fake_img,fakelabels],dim=1).to(device)
            critic_real = critico(real).reshape(-1)
            critic_fake = critico(fake).reshape(-1)
            gp = GP(critico, real, fake)
            loss_critic=(-(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp)
            opt_critico.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critico.step()
        # Calcular Perdida del Generador
        gen_fake = critico(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_generador.zero_grad()
        loss_gen.backward()
        opt_generador.step()
        return loss_critic, loss_gen

    nEpocas = 101
    print("12. Entrenar por Epocas")
    nMuestras = len(data_loader)
    for epoca in range(1, nEpocas):
        closs=0
        gloss=0
        i = 0
        for _,_,onehots,img_and_labels in data_loader:
            i = i + 1
            print(f"item: {i} - bucles: {nMuestras} - epoca: {epoca}")
            loss_critic, loss_gen = entrenarEpoca(onehots, img_and_labels,epoca)   
            closs+=loss_critic.detach()/len(data_loader)
            gloss+=loss_gen.detach()/len(data_loader)
        print(f"Epoca: {epoca}, Perdida Critico: {closs}, Perdida Generador: {gloss}")
        plotearEpoca(epoca)
        scripted = torch.jit.script(generador)
        archivo = "preentrenados/GAN/Lentes/cGAN_Lentes_" + str(epoca) + "_" + str(gloss.item()) + ".pt"
        scripted.save(archivo)

    horaFin = datetime.now()
    tiempoSeg = (horaFin - horaInicio).total_seconds()
    print(f"Se creo y entreno el Modelo cGAN de Lentes en {tiempoSeg} seg")

if __name__ == "__main__":
    mp.freeze_support()
    main()