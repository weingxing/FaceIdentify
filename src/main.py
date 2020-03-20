import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from centerface import CenterFace
# from face_encoder import Encoder


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # self.encoder = Encoder()
        self.centerface = CenterFace(landmarks=True)
        # 初始化定时器
        self.timer_camera = QtCore.QTimer()
        # 初始化摄像头
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.image = None

    def set_ui(self):
        # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_name = QtWidgets.QHBoxLayout()

        self.__layout_fun_label = QtWidgets.QHBoxLayout()
        # QVBoxLayout类垂直地摆放小部件
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.__layout_info_show = QtWidgets.QVBoxLayout()

        self.name_label = QtWidgets.QLabel('姓名：')
        self.name = QtWidgets.QLabel()
        self.button_open_camera = QtWidgets.QPushButton(u'开始识别')
        self.button_close = QtWidgets.QPushButton(u'退出')

        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        # 移动窗口在屏幕上的位置到x = 500，y = 200的位置
        self.move(500, 200)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_name.addWidget(self.name_label, alignment=QtCore.Qt.AlignRight)
        self.__layout_name.addWidget(self.name, alignment=QtCore.Qt.AlignLeft)
        # self.__layout_name.
        self.name.setText('123')

        self.__layout_info_show.addLayout(self.__layout_name)
        self.__layout_info_show.addLayout(self.__layout_data_show)
        self.__layout_info_show.addLayout(self.__layout_fun_button)

        self.__layout_main.addLayout(self.__layout_info_show)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'人脸识别')

    # 建立通信连接
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检查摄像头是否正常',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'结束识别')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'开始识别')

    def show_camera(self):
        # 捕获图像
        flag, self.image = self.cap.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # CenterFace人脸检测
        h, w = self.image.shape[:2]
        dets, lms = self.centerface(self.image, h, w, threshold=0.35)
        # 人脸关键点对齐
        if len(lms) != 0:
            # 提取当前检测到的人脸的特征量
            # embedding = self.encoder.generate_embedding(self.image)
            # 获取数据库中的全部特征量

            # 计算距离，选出最小的距离

            # 判断是否小于最大距离，小于则认证成功
            print(1)
        else:
            print("没有发现人脸")
        # 方框标出人脸用于展示
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(self.image, (int(boxes[0]), int(boxes[1])),
                          (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        show = cv2.resize(self.image, (640, 480))

        # 转换为Qt可以使用的图片格式
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # 将图片更新到界面
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
