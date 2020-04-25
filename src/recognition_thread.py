from core.load_model import load_model
import tensorflow as tf
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from core.face_alignment import Alignment
from core.face_recognition import Recognition

recognition = Recognition()
align = Alignment()
model = './core/models/facenet/facenet.pb'


class Thread(QThread):
    signal = pyqtSignal(str, int)
    flag = False

    def __init__(self):
        super(Thread, self).__init__()
        self.image = None

    def __del__(self):
        pass

    def set_image(self, image):
        self.image = image

    def pretreatment(self, x):
        # mean = np.mean(x)
        # std = np.std(x)
        # std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        # y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        y = self.sess.run(tf.image.per_image_standardization(x))
        return y

    def generate_features(self, image):
        # 取得输入层和输出层的张量及训练标志位
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        pretreatment_face = self.pretreatment(image)
        # 计算特征量
        feed_dict = {images_placeholder: [pretreatment_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def run(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            load_model(model)
        while self.flag:
            if self.image is not None:
                # 提取当前检测到的人脸的特征量
                aligned_img = align.align_face(self.image)
                try:
                    emb = self.generate_features(aligned_img)
                    name, distance = recognition.result(emb)
                    # print(name, distance)
                except Exception as e:
                    name = 'Error'
                    distance = 0
                    print(e)
                self.signal.emit(name, distance)
            self.sleep(1)
