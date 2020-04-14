import sys
import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import *

from view.main_ui import MainWindow
from view.add_ui import AddWindow

from core.centerface import CenterFace
from core.face_recognition import Recognition
from core.face_alignment import Alignment

recognition = Recognition()
align = Alignment()
CAM_NUM = 0


class FirstWindow(MainWindow):
    def __init__(self):
        super().__init__()
        # 人脸识别最大距离
        self.threshold = 0.9
        self.cap = cv2.VideoCapture()
        self.centerface = CenterFace(landmarks=True)
        self.slot_init()
        self.image = None

    # 建立通信连接
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.recognition_timer.timeout.connect(self.identify)
        self.button_add.clicked.connect(self.add)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(CAM_NUM)
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
            aligned_img = align.align_face(self.image)
            # name, distance = self.recognition.result(self.image)
            name, distance = recognition.result(aligned_img)
            print('姓名：%s，距离：%s' % (name, distance))
            if distance < self.threshold:
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
        # 对副本进行修改
        image = self.image.copy()
        h, w = image.shape[:2]
        dets, lms = self.centerface(image, h, w, threshold=0.35)

        # 方框标出人脸用于展示
        # 只取距离摄像头最近的作为识别对象
        if len(dets) != 0:
            det = dets[0]
            boxes, score = det[:4], det[4]
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])),
                          (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        show = cv2.resize(image, (640, 480))

        # 转换为Qt可以使用的图片格式
        show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # 将图片更新到界面
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))

    def add(self):
        if self.cap.isOpened():
            self.timer_camera.stop()
            self.recognition_timer.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.name.setText('未开始识别')
            self.button_open_camera.setText(u'开始识别')
        self.hide()
        self.s = SecondWindow()
        self.s.show()


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


class SecondWindow(AddWindow):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture()
        self.centerface = CenterFace(landmarks=True)
        self.slot_init()
        self.image = None

    # 建立通信连接
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_insert.clicked.connect(self.add_face_id)
        self.button_close.clicked.connect(self.back)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(CAM_NUM)
            if flag:
                self.timer_camera.start(60)
                self.button_open_camera.setText(u'拍摄')
            else:
                QMessageBox.information(self, '错误', '请检查摄像头连接')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.button_open_camera.setText(u'打开相机')

    # 识别身份
    def add_face_id(self):
        name = self.name.text()
        if self.timer_camera.isActive() or self.image is None:
            QMessageBox.information(self, '提示', '请先拍摄照片')
            return
        elif name == '':
            QMessageBox.information(self, '提示', '请输入姓名')
            return
        h, w = self.image.shape[:2]
        dets, lms = self.centerface(self.image, h, w, threshold=0.35)
        if len(lms) != 0:
            # 对齐后提取当前检测到的人脸的特征量
            aligned_img = align.align_face(self.image)
            recognition.add_to_db(name, aligned_img)
            QMessageBox.information(self, '提示', '录入成功')
        else:
            QMessageBox.information(self, '错误', '没有检测到人脸,请重新拍摄')

    # 检测人脸
    def show_camera(self):
        # 捕获图像
        flag, self.image = self.cap.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # 取副本进行修改,保持self.image原始性
        image = self.image.copy()
        # CenterFace人脸检测
        h, w = image.shape[:2]
        dets, lms = self.centerface(image, h, w, threshold=0.35)
        # 方框标出人脸用于展示
        # 只取距离摄像头最近的作为识别对象
        if len(dets) != 0:
            det = dets[0]
            boxes, score = det[:4], det[4]
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])),
                          (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        show = cv2.resize(image, (640, 480))
        # 转换为Qt可以使用的图片格式
        show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # 将图片更新到界面
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_image))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning,
                                    u'关闭', u'是否关闭本窗口?')
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

    def back(self):
        if self.cap.isOpened():
            self.timer_camera.stop()
            self.cap.release()
            self.button_open_camera.setText(u'打开相机')
        self.hide()
        self.f = FirstWindow()
        self.f.show()


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = FirstWindow()
    window.show()
    sys.exit(App.exec_())
