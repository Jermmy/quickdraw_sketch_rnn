import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QPen, QImage, qRgb
from PyQt5.QtCore import *


class SketchArea(QWidget):

    newSketchDown = pyqtSignal()
    clearSketchDown = pyqtSignal()

    def __init__(self):
        super().__init__()
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setSizePolicy(sizePolicy)
        self.setFocusPolicy(Qt.StrongFocus)

        self.lastPoint = QPoint()
        # 判断鼠标触摸点在SketchArea内
        self.scribbling = False
        # 画笔
        self.penWidth = 3
        self.penColor = Qt.black
        #self.pen = QPen(Qt.black, self.penWidth, Qt.SolidLine)

        self.image = QImage(500, 500, QImage.Format_RGB32)
        self.image.fill(qRgb(255, 255, 255))

        self.sketch = []
        self.stroke = []

        self.initUi()

    def initUi(self):
        self.setWindowTitle("Sketch Area")
        self.setGeometry(300, 300, 500, 500)



    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == Qt.Key_C:
            self.clearImage()
            self.clearSketchDown.emit()

            self.sketch = []
            self.stroke = []
        else:
            # QWidget.keyPressEvent(QKeyEvent)
            return


    def keyReleaseEvent(self, QKeyEvent):
        QWidget.keyReleaseEvent(self, QKeyEvent)


    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.lastPoint = QMouseEvent.pos()
            self.scribbling = True
            self.stroke.append([QMouseEvent.pos().x(), QMouseEvent.pos().y()])
        else:
            QWidget.keyPressEvent(self, QMouseEvent)


    def mouseMoveEvent(self, QMouseEvent):
        if QMouseEvent.buttons() & Qt.LeftButton and self.scribbling:
            self.drawLineTo(QMouseEvent.pos())
            self.stroke.append([QMouseEvent.pos().x(), QMouseEvent.pos().y()])
        else:
            QWidget.keyPressEvent(self, QMouseEvent)


    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(QMouseEvent.pos())
            self.scribbling = False
            self.sketch.append(self.stroke)
            self.stroke = []

        self.newSketchDown.emit()



    def paintEvent(self, QPaintEvent):
        qp = QPainter(self)
        dirtyRect = QPaintEvent.rect()
        qp.drawImage(dirtyRect, self.image, dirtyRect)


    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.update()


    # def resizeEvent(self, QResizeEvent):
    #     if self.width() > self.image.width() or self.height() > self.image.height():
    #         newWidth = max(self.width() + 128, self.image.width())
    #         newHeight = max(self.height() + 128, self.image.height())
    #         size = QSize(newWidth, newHeight)
    #         # self.resizeImage(size)
    #         self.update()


    def drawLineTo(self, point):
        qp = QPainter(self.image)

        qp.setRenderHint(QPainter.Antialiasing)

        qp.setPen(QPen(self.penColor, self.penWidth, Qt.SolidLine,
                       Qt.RoundCap, Qt.RoundJoin))

        if point.x() < self.rect().width() and point.y() < self.rect().height():
            qp.drawLine(self.lastPoint, point)

        rad = int(self.penWidth / 2 + 2)

        self.update(QRect(self.lastPoint, point).normalized().adjusted(-rad, -rad, rad, rad))

        self.lastPoint = point


if __name__ == '__main__':
    app = QApplication(sys.argv)
    s = SketchArea()
    s.show()
    sys.exit(app.exec_())
