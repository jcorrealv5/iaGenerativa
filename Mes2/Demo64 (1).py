import cv2, os
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

print("Demo 64: Cambio de Sonrisa en Tiempo Real")
rutaSonrisa = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSonrisa/"
rutaConrisa = rutaSonrisa + "Si"
archivoConrisa = os.path.join(rutaConrisa, "Luis.jpg")
imgConrisa = np.array(Image.open(archivoConrisa).convert("RGB"))

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
gen = Generador(img_channels=3, num_residuals=9).to(device)
gen.load_state_dict(torch.load("preentrenados/GAN/Sonrisa/gen_conrisa_10.pth"))

video = cv2.VideoCapture(0)
if(video.isOpened()):
    while(True):
        rpta, imgOriginal = video.read()
        if(rpta):
            img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256,256))
            imgRisa = imgConrisa.copy()
            augmentations = transforms(image=imgRisa, image0=img)
            imgSinrisa = augmentations["image0"]
            imgConrisa = augmentations["image"]
            dstSonrisa = [(imgSinrisa, imgConrisa)]
            loaderSonrisa=torch.utils.data.DataLoader(dstSonrisa,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            for sinrisa,conrisa in loaderSonrisa:
                fakeRisa=gen(sinrisa.to(device)).squeeze(0)
            imgConrisa = fakeRisa/2+0.5
            imgConrisa = (imgConrisa.clamp(0, 1) * 255).byte()
            imgConrisa = imgConrisa.permute(1,2,0).cpu().numpy()
            imgOriginal = cv2.resize(imgOriginal, (256,256))
            imgConrisa = cv2.resize(imgConrisa, (imgOriginal.shape[1],imgOriginal.shape[0]))
            imgConrisa = cv2.cvtColor(imgConrisa, cv2.COLOR_RGB2BGR)
            imagenes = cv2.hconcat([imgOriginal, imgConrisa])
            cv2.imshow("Sonrisa", imagenes)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No esta activa la camara")