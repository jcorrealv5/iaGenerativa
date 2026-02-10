import sys, os, cv2, time
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import numpy as np
import torch, torchvision
import torchvision.transforms as T

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgImagenAritmetica.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtNroVeces = self.findChild(QtWidgets.QSpinBox, "txtNroVeces")
        self.cboTipoTransicion = self.findChild(QtWidgets.QComboBox, "cboTipoTransicion")
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
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        #Llenar el Combo de Tipo de Transicion
        self.cboTipoTransicion.addItem("Con a Sin Lentes")
        self.cboTipoTransicion.addItem("Sin a Con Lentes")
        
    def procesarImagen(self):
        self.nVeces = self.txtNroVeces.value()
        self.tipoTransicion = self.cboTipoTransicion.currentIndex()
        thread = WorkerModeloGAN(self)
        thread.inicio.connect(self.mostrarInicio)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarInicio(self, imagenes, titulos):
        contImagenes = [self.lblImagen1, self.lblImagen2, self.lblImagen3]
        contLabels = [self.lblMensaje1, self.lblMensaje2, self.lblMensaje3]
        for i in range(3):
            img = imagenes[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (alto,ancho) = img.shape[:2]
            qImg = QImage(img.data, ancho, alto, ancho*3, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qImg)
            contImagenes[i].setPixmap(pix)
            contLabels[i].setText(titulos[i])

    def mostrarProgreso(self, n, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (alto,ancho) = img.shape[:2]
        qImg = QImage(img.data, ancho, alto, ancho*3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.lblImagen4.setPixmap(pix)
        self.lblMensaje4.setText(f"{n} de {self.nVeces}")

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloGAN(QThread):
    inicio = QtCore.pyqtSignal(list, list)
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, np.ndarray)
    
    def __init__(self, parent):
        super(WorkerModeloGAN, self).__init__(parent)
        self.nVeces = parent.nVeces
        self.tipoTransicion = parent.tipoTransicion

    def generarRuidoEtiqueta(self, indice):
        noise_g = torch.randn(1, 100, 1, 1)
        labels_g = torch.zeros(1, 2, 1, 1)
        labels_g[:,indice,:,:]=1
        noise_and_labels=torch.cat([noise_g,labels_g],dim=1).to(self.device)
        imagenesGeneradas=self.generador(noise_and_labels)
        return imagenesGeneradas[0],noise_g, labels_g

    def convertirTensorToArray(self, imagenes):
        imgs = []
        for i in range(len(imagenes)):
            img = imagenes[i]/2+0.5 #-1,1
            img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
            img = img.permute(1,2,0).cpu().numpy()
            imgs.append(img)
        return imgs
    
    def run(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        archivo = "preentrenados/GAN/Lentes/cGAN_Lentes_100_1077.6683349609375.pt"
        self.generador=torch.jit.load(archivo, map_location=self.device)
        self.generador.eval()
        #Parte 1: Generar las 3 caras de referencia
        imgConLentes,ruidoConLentes,labelConLentes = self.generarRuidoEtiqueta(0)
        imgSinLentes,ruidoSinLentes,labelSinLentes = self.generarRuidoEtiqueta(1)
        imgPersona,ruidoPersona,labelPersona = self.generarRuidoEtiqueta(1)        
        imgs = self.convertirTensorToArray([imgConLentes, imgSinLentes, imgPersona])
        titulos = ["Con Lentes", "Sin Lentes", "Referencia"]
        self.inicio.emit(imgs, titulos)
        #Parte 2: Generar las n caras de transicion
        pesos = np.linspace(0,1,self.nVeces)
        for i in range(self.nVeces):
            if(self.tipoTransicion==0):
                label=pesos[i]*labelSinLentes+(1-pesos[i])*labelConLentes
            else:
                label=pesos[i]*labelConLentes+(1-pesos[i])*labelSinLentes
            noise_and_labels=torch.cat([ruidoPersona.reshape(1, 100, 1, 1),label.reshape(1, 2, 1, 1)],dim=1).to(self.device) 
            fakes=self.generador(noise_and_labels)
            img = self.convertirTensorToArray(fakes)[0]
            time.sleep(0.5)
            self.progreso.emit(i+1,img)
        rpta = "Se generaron caras"
        self.finalizado.emit(rpta)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())