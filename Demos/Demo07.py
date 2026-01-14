import sys, os, cv2
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgVisorImagen.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtArchivo = self.findChild(QtWidgets.QLineEdit, "txtArchivo")
        btnAbrirArchivo = self.findChild(QtWidgets.QPushButton, "btnAbrirArchivo")
        btnNuevaImagen = self.findChild(QtWidgets.QPushButton, "btnNuevaImagen")
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblCaraOriginal = self.findChild(QtWidgets.QLabel, "lblCaraOriginal")
        self.lblCaraGenerada = self.findChild(QtWidgets.QLabel, "lblCaraGenerada")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        #Programar los eventos clicks de los Botones
        btnAbrirArchivo.clicked.connect(self.abrirArchivo)
        btnNuevaImagen.clicked.connect(self.nuevaImagen)
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        
    def abrirArchivo(self):
        dlg= QFileDialog()
        dlg.exec()
        self.archivo = dlg.selectedFiles()[0]
        self.txtArchivo.setText(self.archivo)
        pix = QPixmap(self.archivo)
        self.lblCaraOriginal.setPixmap(pix)

    def nuevaImagen(self):
        self.txtArchivo.setText("")
        self.lblCaraOriginal.setPixmap(QPixmap())
        self.lblCaraGenerada.setPixmap(QPixmap())
        self.lblMensaje.setText("")

    def procesarImagen(self):
        thread = WorkerModeloAE(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, item, archivo, imgReconstruida):
        self.lblMensaje.setText(f"{item + 1} - {archivo}")
        #Convertir el Tensor generado en una imagen o array de NumPy RGB
        img_tensor = imgReconstruida.reshape((3, 100, 100))
        img_array = np.transpose(img_tensor.detach().cpu().numpy(), (1, 2, 0))
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)               
        img_final = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        #Mostrar la imagen generada en el control Label
        (alto,ancho) = img_final.shape[:2]
        qImg = QImage(img_final.data, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.lblCaraGenerada.setPixmap(pix)
        #Grabar la imagen generada a disco
        objRuta = Path(self.txtArchivo.text())
        categoria = objRuta.parent.name
        rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/Alumnos_AE/" + categoria
        if(not os.path.isdir(rutaDestino)):
            os.makedirs(rutaDestino)
        archivo = os.path.join(rutaDestino, "AE_" + categoria + "_" + str(item+1).rjust(3, '0') + ".png")
        cv2.imwrite(archivo, img_final)

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloAE(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, str, torch.Tensor)
    
    def __init__(self, parent):
        super(WorkerModeloAE, self).__init__(parent)
        self.archivo = parent.archivo
        self.rutaPT = "C:/Data/Python/2026_01_IAG/Demos/carasAE"
        self.archivosPT = os.listdir(self.rutaPT)
        self.archivosPT.sort(reverse=True)
    
    def run(self):
        nArchivosPT = len(self.archivosPT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transform = T.Compose([T.ToTensor(), T.Resize(100)])
        imagen = Image.open(self.archivo).convert("RGB")
        imagenTensor = transform(imagen).unsqueeze(0)
        for i in range(nArchivosPT):
            modelo=torch.jit.load(os.path.join(self.rutaPT, self.archivosPT[i]),map_location=device)
            modelo.eval()
            with torch.no_grad():
                imagenPlana = imagenTensor.reshape((1,30000))
                imgReconstruida, mu = modelo(imagenPlana.to(device))
                self.progreso.emit(i, self.archivosPT[i], imgReconstruida)
        rpta = "Finalizo de Procesar las imagenes"
        self.finalizado.emit(rpta)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())