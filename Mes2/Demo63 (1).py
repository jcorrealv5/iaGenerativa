import cv2, os
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

print("Demo 62: Cambio de Sexo a Femenino en Tiempo Real")
rutaSexo = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSexo/"
rutaSexoMujer = rutaSexo + "Femenino"
archivoSexoMujer = os.path.join(rutaSexoMujer, "Dua_Lipa.jpg")
imgSexoMujer = np.array(Image.open(archivoSexoMujer).convert("RGB"))

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
gen = Generador(img_channels=3, num_residuals=9).to(device)
gen.load_state_dict(torch.load("preentrenados/GAN/Sexo/gen_mujer_20.pth"))

video = cv2.VideoCapture(0)
if(video.isOpened()):
    while(True):
        rpta, imgOriginal = video.read()
        if(rpta):
            img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256,256))
            imgMujer = imgSexoMujer.copy()
            augmentations = transforms(image=imgMujer, image0=img)
            imgMujer = augmentations["image0"]
            imgHombre = augmentations["image"]
            dstSexo = [(imgMujer, imgHombre)]
            loaderSexo=torch.utils.data.DataLoader(dstSexo,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            for mujer,hombre in loaderPelo:
                fakeMujer=gen(hombre.to(device)).squeeze(0)
            imgMujer = fakeMujer/2+0.5
            imgMujer = (imgMujer.clamp(0, 1) * 255).byte()
            imgMujer = imgMujer.permute(1,2,0).cpu().numpy()
            imgOriginal = cv2.resize(imgOriginal, (256,256))
            imgMujer = cv2.resize(imgMujer, (imgOriginal.shape[1],imgOriginal.shape[0]))
            imgMujer = cv2.cvtColor(imgMujer, cv2.COLOR_RGB2BGR)
            imagenes = cv2.hconcat([imgOriginal, imgMujer])
            cv2.imshow("Sexo", imagenes)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No esta activa la camara")