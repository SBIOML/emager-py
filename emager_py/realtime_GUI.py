from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import os

class ui(QWidget):
    def __init__(self, nb_class):

        QWidget.__init__(self)

        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        file_path = os.path.join(current_dir_path, 'data.txt')

        if nb_class == 4:
            self.imgPath = os.path.join(current_dir_path, 'Fourgesture')
            self.label = [0, 1, 2, 3]
        if nb_class == 5:
            self.imgPath = os.path.join(current_dir_path, 'Fivegesture')
            self.label = [0, 1, 2, 3, 4]
        if nb_class == 6:
            self.imgPath = os.path.join(current_dir_path, 'OBgesture')
            self.label = [0, 1, 2, 3, 4, 5]
        self.pixmaps = [QPixmap(self.imgPath + str(self.label[i]) + '.jpg') for i in self.label]
        self.pixmaps = [pm.scaled(QSize(400, 400)) for pm in self.pixmaps]
        self.n = 0
        self.setWindowTitle('RealTime Gesture Recognition')

        layout = QGridLayout()
        self.gestureImage = QtWidgets.QLabel(self) # alignment=Qt.AlignCenter
        self.gestureImage.setPixmap(self.pixmaps[0])
        layout.addWidget(self.gestureImage, 0, 0)

        self.setLayout(layout)

    def setImg(self, label):
        self.gestureImage.setPixmap(self.pixmaps[label])
        self.gestureImage.repaint()
        #print("Label on screen:_____",label)


if __name__ == '__main__':
    app = QApplication([])
    ui = ui()
    ui.show()
    timer = QTimer()
    timer.timeout.connect(lambda : ui.setImg(4))
    timer.start(1000)
    app.exec()