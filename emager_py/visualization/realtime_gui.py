from PyQt6 import QtWidgets
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
import os
import sys
import threading

class RealTimeGestureUi(QWidget):
    labelChanged = pyqtSignal(int)  # Define a signal for changing labels

    def __init__(self, images:list):
        self.app = QApplication([])

        super().__init__()

        self.images_path = images
        self.img_label = 0
        self.label_text = "Label Text"
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.setImg(self.img_label))
        

        self.pixmaps = [QPixmap(img) for img in self.images_path]   
        self.pixmaps = [pm.scaled(QSize(400, 400)) for pm in self.pixmaps]
        self.setWindowTitle('RealTime Gesture Recognition')

        layout = QGridLayout()

        self.labelText = QtWidgets.QLabel(self, alignment=Qt.AlignmentFlag.AlignCenter)
        self.labelText.setText(self.label_text)
        layout.addWidget(self.labelText, 0, 0)

        self.gestureImage = QtWidgets.QLabel(self)  # alignment=Qt.AlignCenter
        self.gestureImage.setPixmap(self.pixmaps[self.img_label])
        layout.addWidget(self.gestureImage, 1, 0)

        self.setLayout(layout)

        self.labelChanged.connect(self.setImg)

    @pyqtSlot(int)
    def setImg(self, label):
        self.img_label = label
        self.label_text = self.images_path[label].split("/")[-1].split(".")[0]
        self.label_text = f"{self.label_text} (label : {label})"
        self.labelText.setText(self.label_text)
        self.gestureImage.setPixmap(self.pixmaps[label])

    def update_label(self, label):
        self.labelChanged.emit(label)

    def run(self):
        self.show()
        self.timer.start(1000)
        self.app.aboutToQuit.connect(self.stop)
        self.app.exec()

    def stop(self):
        self.timer.stop()
