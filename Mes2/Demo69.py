import cv2, os
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

print("Demo 69: Poner Bigote a los que No lo tienen")
rutaBigote = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestBigote/"
rutaConBigote = rutaBigote + "Si"
archivoBigote = os.path.join(rutaConBigote, "Mayimbu.jpeg")
imgConBigote = np.array(Image.open(archivoBigote).convert("RGB"))

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
gen = Generador(img_channels=3, num_residuals=9).to(device)
gen.load_state_dict(torch.load("preentrenados/GAN/Bigote/gen_conbigote_6.pth"))

video = cv2.VideoCapture(0, 700)
if(video.isOpened()):
    while(True):
        rpta, imgOriginal = video.read()
        if(rpta):
            img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256,256))
            imgBigote = imgConBigote.copy()
            augmentations = transforms(image=imgBigote, image0=img)
            imgBigote = augmentations["image0"]
            imgNormal = augmentations["image"]
            dstBigote = [(imgNormal, imgBigote)]
            loaderBigote=torch.utils.data.DataLoader(dstBigote,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
            for sinbigote,conbigote in loaderBigote:
                fakeBigote=gen(sinbigote.to(device)).squeeze(0)
            imgBigote = fakeBigote/2+0.5
            imgBigote = (imgBigote.clamp(0, 1) * 255).byte()
            imgBigote = imgBigote.permute(1,2,0).cpu().numpy()
            imgOriginal = cv2.resize(imgOriginal, (256,256))
            imgBigote = cv2.resize(imgBigote, (imgOriginal.shape[1],imgOriginal.shape[0]))
            imgBigote = cv2.cvtColor(imgBigote, cv2.COLOR_RGB2BGR)
            imagenes = cv2.hconcat([imgOriginal, imgBigote])
            cv2.imshow("Bigotes", imagenes)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No esta activa la camara")