import sys, os, cv2, time
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgImagenTransformaciones.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtArchivo = self.findChild(QtWidgets.QLineEdit, "txtArchivo")
        btnAbrirImagen = self.findChild(QtWidgets.QPushButton, "btnAbrirImagen")
        self.cboSexo = self.findChild(QtWidgets.QComboBox, "cboSexo")
        self.cboPelo = self.findChild(QtWidgets.QComboBox, "cboPelo")
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblImagen1 = self.findChild(QtWidgets.QLabel, "lblImagen1")
        self.lblMensaje1 = self.findChild(QtWidgets.QLabel, "lblMensaje1")
        self.lblImagen2 = self.findChild(QtWidgets.QLabel, "lblImagen2")
        self.lblMensaje2 = self.findChild(QtWidgets.QLabel, "lblMensaje2")
        self.lblImagen3 = self.findChild(QtWidgets.QLabel, "lblImagen3")
        self.lblMensaje3 = self.findChild(QtWidgets.QLabel, "lblMensaje3")
        self.lblImagen4 = self.findChild(QtWidgets.QLabel, "lblImagen4")
        self.lblMensaje4 = self.findChild(QtWidgets.QLabel, "lblMensaje4")
        #Programar los eventos clicks de los Botones
        btnAbrirImagen.clicked.connect(self.abrirImagen)
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        #Llenar el Combo de Sexo
        self.sexos = ["Mujer", "Hombre"]
        self.cboSexo.addItems(self.sexos)
        #Llenar el Combo de Pelo
        self.pelos = ["Negro", "Rubio"]
        self.cboPelo.addItems(self.pelos)

    def abrirImagen(self):
        dlg = QFileDialog()
        dlg.setDirectory("C:/Data/Python/2026_01_IAG/Demos/datasets")
        dlg.setNameFilter("Imagenes (*.png *.jpg)");
        dlg.exec()
        self.archivo = dlg.selectedFiles()[0]
        self.txtArchivo.setText(self.archivo)
        pix = QPixmap(self.archivo)
        self.lblImagen1.setPixmap(pix)

    def procesarImagen(self):
        self.sexo = self.cboSexo.currentIndex()
        self.pelo = self.cboPelo.currentIndex()
        self.lblMensaje1.setText(f"{self.sexos[self.sexo]} de Pelo {self.pelos[self.pelo]}")
        self.lblMensaje2.setText(f"{self.sexos[self.sexo]} de Pelo {self.pelos[1-self.pelo]}")
        self.lblMensaje3.setText(f"{self.sexos[1-self.sexo]} de Pelo {self.pelos[self.pelo]}")
        self.lblMensaje4.setText(f"{self.sexos[1-self.sexo]} de Pelo {self.pelos[1-self.pelo]}")
        rutaPelo = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestPelo/"
        rutaPeloOpuesto = rutaPelo + ("Blond" if self.pelo==0 else "Black")
        archivoPeloOpuesto = os.path.join(rutaPeloOpuesto, "B0.png" if self.pelo==0 else "A0.png")
        rutaSexo = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSexo/"
        rutaSexoOpuesto = rutaSexo + ("Masculino" if self.sexo==0 else "Femenino")
        archivoSexoOpuesto = os.path.join(rutaSexoOpuesto, "Brad_Pitt.jpg" if self.sexo==0 else "Ana.jpeg")
        self.img_Original = np.array(Image.open(self.archivo).convert("RGB"))
        self.img_PeloOpuesto = np.array(Image.open(archivoPeloOpuesto).convert("RGB"))
        self.img_SexoOpuesto = np.array(Image.open(archivoSexoOpuesto).convert("RGB"))        
        thread = WorkerModeloGAN(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.start()

    def mostrarRpta(self, imagenes):
        contImagenes = [self.lblImagen2, self.lblImagen3, self.lblImagen4]
        for i in range(3):
            img = imagenes[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (alto,ancho) = img.shape[:2]
            qImg = QImage(img.data, ancho, alto, ancho*3, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qImg)
            contImagenes[i].setPixmap(pix)

class WorkerModeloGAN(QThread):
    finalizado = QtCore.pyqtSignal(list)
    
    def __init__(self, parent):
        super(WorkerModeloGAN, self).__init__(parent)
        self.pelo = parent.pelo
        self.sexo = parent.sexo
        self.img_Original = parent.img_Original
        self.img_PeloOpuesto = parent.img_PeloOpuesto
        self.img_SexoOpuesto = parent.img_SexoOpuesto

    def convertirTensorToArray(self, imagenesTensor):
        imgs = []
        for i in range(len(imagenesTensor)):
            img = imagenesTensor[i]/2+0.5 #-1,1
            img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
            img = img.permute(1,2,0).cpu().numpy()
            imgs.append(img)
        return imgs

    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 1
        transforms = albumentations.Compose(
        [albumentations.Resize(width=256, height=256),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],max_pixel_value=255),
            ToTensorV2()],
        additional_targets={"image0": "image"})
        #Procesar el Modelo del Pelo
        if(self.pelo==1):
            augmentations = transforms(image=self.img_Original, image0=self.img_PeloOpuesto)
        else:
            augmentations = transforms(image=self.img_PeloOpuesto, image0=self.img_Original)
        imgPeloNegro = augmentations["image0"]
        imgPeloRubio = augmentations["image"]
        dstPelo = [(imgPeloNegro, imgPeloRubio)]
        loaderPelo=torch.utils.data.DataLoader(dstPelo,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        gen_A = Generador(img_channels=3, num_residuals=9).to(device)
        gen_B = Generador(img_channels=3, num_residuals=9).to(device)
        gen_A.load_state_dict(torch.load("preentrenados/GAN/CelebA/gen_black.pth"))
        gen_B.load_state_dict(torch.load("preentrenados/GAN/CelebA/gen_blond.pth"))
        for black,blond in loaderPelo:
            if(self.pelo==0):
                fakePelo=gen_B(black.to(device)).squeeze(0)
            else:
                fakePelo=gen_A(blond.to(device)).squeeze(0)        
        #Procesar el Modelo del Sexo
        if(self.sexo==0):
            augmentations = transforms(image=self.img_Original, image0=self.img_SexoOpuesto)
        else:
            augmentations = transforms(image=self.img_SexoOpuesto, image0=self.img_Original)
        imgSexoMujer = augmentations["image0"]
        imgSexoHombre = augmentations["image"]
        dstSexo = [(imgSexoMujer, imgSexoHombre)]
        loaderSexo=torch.utils.data.DataLoader(dstSexo,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        gen_A = Generador(img_channels=3, num_residuals=9).to(device)
        gen_B = Generador(img_channels=3, num_residuals=9).to(device)
        gen_A.load_state_dict(torch.load("preentrenados/GAN/Sexo/gen_mujer_20.pth"))
        gen_B.load_state_dict(torch.load("preentrenados/GAN/Sexo/gen_hombre_20.pth"))
        for mujer,hombre in loaderSexo:
            if(self.sexo==1):
                fakeSexo=gen_A(mujer.to(device)).squeeze(0)
            else:
                fakeSexo=gen_B(hombre.to(device)).squeeze(0)
        imgFakeSexo = fakeSexo.permute(1,2,0).cpu().detach().numpy()
        #Procesar el Modelo del Pelo al Sexo Cambiado
        if(self.pelo==1):
            augmentations = transforms(image=imgFakeSexo, image0=self.img_PeloOpuesto)
        else:
            augmentations = transforms(image=self.img_PeloOpuesto, image0=imgFakeSexo)
        imgSexoPeloNegro = augmentations["image0"]
        imgSexoPeloRubio = augmentations["image"]
        dstSexoPelo = [(imgSexoPeloNegro, imgSexoPeloRubio)]
        loaderSexoPelo=torch.utils.data.DataLoader(dstSexoPelo,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        gen_A = Generador(img_channels=3, num_residuals=9).to(device)
        gen_B = Generador(img_channels=3, num_residuals=9).to(device)
        gen_A.load_state_dict(torch.load("preentrenados/GAN/CelebA/gen_black.pth"))
        gen_B.load_state_dict(torch.load("preentrenados/GAN/CelebA/gen_blond.pth"))
        for black,blond in loaderSexoPelo:
            if(self.pelo==0):
                fakeSexoPelo=gen_B(black.to(device)).squeeze(0)
            else:
                fakeSexoPelo=gen_A(blond.to(device)).squeeze(0)
        imagenesTensor = [fakePelo, fakeSexo, fakeSexoPelo]
        imagenesArray = self.convertirTensorToArray(imagenesTensor)
        self.finalizado.emit(imagenesArray)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())