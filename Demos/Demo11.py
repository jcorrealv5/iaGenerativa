import sys, os, cv2
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
        uic.loadUi("dlgImagen.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        #Programar los eventos clicks de los Botones
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        
    def procesarImagen(self):
        thread = WorkerModeloVAE(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, n, imgReconstruida):
        #Convertir el Tensor generado en una imagen o array de NumPy RGB
        img_array = imgReconstruida.reshape((28, 28)).numpy()         
        img_final = (img_array * 255).clip(0, 255).astype(np.uint8)
        #Mostrar la imagen generada en el control Label
        (alto,ancho) = img_final.shape[:2]
        qImg = QImage(img_final.data, ancho, alto, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qImg)
        self.lblImagen.setPixmap(pix)
        total = 21*21
        self.lblMensaje.setText(f"{n} de {total}")

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloVAE(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, torch.Tensor)
    
    def __init__(self, parent):
        super(WorkerModeloVAE, self).__init__(parent)
    
    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo=torch.jit.load('preentrenados/VAE/MNIST/VAE_Digitos_99_130.20439571940105.pt',map_location=device)
        transformacion_data = T.Compose([T.ToTensor()])
        dataset_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
        batchSize = 100
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=True)
        y = -1.0
        c = 0
        for i in range(21):
            x = -1.0            
            for j in range(21):
                c=c+1
                z_sample = torch.tensor([[x, y]], dtype=torch.float).to(device)
                x_decoded = modelo.decode(z_sample)
                digitoGenerado = x_decoded.detach().cpu().reshape(28, 28)
                QThread.msleep(50)
                self.progreso.emit(c,digitoGenerado)
                x += 0.1
            y += 0.1
        rpta = "Se generaron {0} digitos".format(21 * 21)
        self.finalizado.emit(rpta)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())