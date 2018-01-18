from .sketcharea import SketchArea

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.sketcharea = SketchArea()
        self.label = QLabel('wwwww')

        self.mainSplitter = QSplitter(Qt.Vertical)
        self.mainSplitter.addWidget(self.label)
        self.mainSplitter.addWidget(self.sketcharea)

        self.setCentralWidget(self.mainSplitter)

        self.setGeometry(300, 300, 500, 600)
        self.setWindowTitle('QuickDraw')

        self.sketcharea.newSketchDown.connect(self.search)
        self.sketcharea.clearSketchDown.connect(self.clear_sketch)

    def search(self):
        pass

    def clear_sketch(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())