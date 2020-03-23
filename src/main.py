import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from centerface import CenterFace
from face_recognition import Recognition
from face_alignment import Alignment


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # 初始化人脸识别、对齐、检测
        self.recognition = Recognition()
        self.align = Alignment()
        self.centerface = CenterFace(landmarks=True)
        # 初始化定时器
        self.timer_camera = QtCore.QTimer()
        self.recognition_timer = QtCore.QTimer()
        # 初始化摄像头
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 1
        # 初始化界面、连接槽函数
        self.set_ui()
        self.slot_init()
        self.image = None

    def set_ui(self):
        # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.layout_main = QtWidgets.QHBoxLayout()
        self.layout_fun_button = QtWidgets.QHBoxLayout()
        self.layout_name = QtWidgets.QHBoxLayout()

        # QVBoxLayout类垂直地摆放小部件
        self.layout_left = QtWidgets.QVBoxLayout()

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

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.layout_fun_button.addWidget(self.button_open_camera)
        self.layout_fun_button.addWidget(self.button_close)

        self.layout_name.addWidget(self.name_label, alignment=QtCore.Qt.AlignRight)
        self.layout_name.addWidget(self.name, alignment=QtCore.Qt.AlignLeft)
        self.name.setText('未开始识别')

        self.layout_left.addLayout(self.layout_name)
        self.layout_left.addLayout(self.layout_fun_button)

        self.layout_main.addLayout(self.layout_left)
        self.layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.layout_main)
        self.setWindowTitle(u'人脸识别')

    # 建立通信连接
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.recognition_timer.timeout.connect(self.identify)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if flag:
                self.timer_camera.start(60)
                self.recognition_timer.start(1000)
                self.button_open_camera.setText(u'结束识别')
            else:
                print('请检查摄像头连接')
        else:
            self.timer_camera.stop()
            self.recognition_timer.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.name.setText('未开始识别')
            self.button_open_camera.setText(u'开始识别')

    # 识别身份
    def identify(self):
        h, w = self.image.shape[:2]
        dets, lms = self.centerface(self.image, h, w, threshold=0.35)
        if len(lms) != 0:
            # 提取当前检测到的人脸的特征量
            aligned_img = self.align.align_face(self.image)
            # name, distance = self.recognition.result(self.image)
            name, distance = self.recognition.result(aligned_img)
            print('姓名：%s，距离：%s' % (name, distance))
            if distance < 0.9:
                self.name.setText(name)
            else:
                self.name.setText("不认识")
        else:
            self.name.setText('没有发现人脸')
            print("没有发现人脸")

    # 检测人脸
    def show_camera(self):
        # 捕获图像
        flag, self.image = self.cap.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # CenterFace人脸检测
        h, w = self.image.shape[:2]
        dets, lms = self.centerface(self.image, h, w, threshold=0.35)

        # 方框标出人脸用于展示
        # 只取距离摄像头最近的作为识别对象
        if len(dets) != 0:
            det = dets[0]
            boxes, score = det[:4], det[4]
            cv2.rectangle(self.image, (int(boxes[0]), int(boxes[1])),
                          (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        show = cv2.resize(self.image, (640, 480))

        # 转换为Qt可以使用的图片格式
        show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # 将图片更新到界面
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))

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
    window = MainWindow()
    window.show()
    sys.exit(App.exec_())
