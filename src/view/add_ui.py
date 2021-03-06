import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *


class AddWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AddWindow, self).__init__(parent)
        # QVBoxLayout类垂直地摆放小部件
        self.layout_left = QtWidgets.QVBoxLayout()
        # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.layout_name = QtWidgets.QHBoxLayout()
        self.layout_fun_button = QtWidgets.QHBoxLayout()
        self.layout_main = QtWidgets.QHBoxLayout()

        self.name_label = QtWidgets.QLabel('姓名：')
        self.name = QtWidgets.QLineEdit()
        self.button_insert = QtWidgets.QPushButton(u'录入')
        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_close = QtWidgets.QPushButton(u'返回')
        self.label_show_camera = QtWidgets.QLabel()
        self.timer_camera = QtCore.QTimer()
        self.set_ui()

    def set_ui(self):
        self.button_open_camera.setMinimumHeight(50)
        self.button_insert.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        # 移动窗口在屏幕上的位置到x = 500，y = 200的位置
        self.move(500, 200)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.layout_fun_button.addWidget(self.button_insert)
        self.layout_fun_button.addWidget(self.button_open_camera)
        self.layout_fun_button.addWidget(self.button_close)

        self.layout_name.addWidget(self.name_label, alignment=QtCore.Qt.AlignRight)
        self.layout_name.addWidget(self.name, alignment=QtCore.Qt.AlignLeft)
        self.layout_left.addLayout(self.layout_name)
        self.layout_left.addLayout(self.layout_fun_button)

        self.layout_main.addLayout(self.layout_left)
        self.layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.layout_main)
        self.setWindowTitle(u'人脸信息录入')


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = AddWindow()
    window.show()
    sys.exit(App.exec_())
