import cv2, os
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

print("Demo 62: Cambio de Pelo a Rubio en Tiempo Real")
rutaPelo = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestPelo/"
rutaPeloRubio = rutaPelo + "Blond"
archivoPeloRubio = os.path.join(rutaPeloRubio, "B0.png")
imgPeloRubio = np.array(Image.open(archivoPeloRubio).convert("RGB"))

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
gen = Generador(img_channels=3, num_residuals=9).to(device)
gen.load_state_dict(torch.load("preentrenados/GAN/Pelos/gen_blond.pth"))

video = cv2.VideoCapture(0)
if(video.isOpened()):
    while(True):
        rpta, img = video.read()
        if(rpta):
            img = cv2.resize(img, (256,256))
            augmentations = transforms(image=imgPeloRubio, image0=img)
            imgPeloNegro = augmentations["image0"]
            imgPeloRubio = augmentations["image"]
            dstPelo = [(imgPeloNegro, imgPeloRubio)]
            loaderPelo=torch.utils.data.DataLoader(dstPelo,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            for black,blond in loaderPelo:
                fakePeloRubio=gen(black.to(device)).squeeze(0)
            imgRubio = fakePeloRubio/2+0.5
            imgRubio = (imgRubio.clamp(0, 1) * 255).byte()
            imgRubio = imgRubio.permute(1,2,0).cpu().numpy()
            imgRubio = cv2.resize(imgRubio, (img.shape[1],img.shape[0]))
            imagenes = cv2.hconcat([img, imgRubio])
            cv2.imshow("Rubio", imagenes)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No esta activa la camara")