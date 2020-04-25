# 获取人脸特征向量
import os
import cv2
import time
import numpy as np
import tensorflow as tf

from core.load_model import load_model

# 模型路径
facenet_model_checkpoint = os.path.abspath("G:/facenet.pb")
# 预训练模型
# facenet_model_checkpoint = os.path.abspath("G:\\facenet_pretrained\\20180408-102900.pb")


class Encoder:
    def __init__(self):
        # self.detection = Detection()
        self.sess = tf.Session()
        start = time.time()
        with self.sess.as_default():
            load_model(facenet_model_checkpoint)
        print('载入模型成功，用时：%ds' % (time.time() - start))

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
        features = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = self.pretreatment(image)
        # 计算特征量
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(features, feed_dict=feed_dict)[0]

    def distance(self, emb1, emb2):
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff))
        dist = np.sum(np.square(emb1 - emb2))
        return dist


if __name__ == '__main__':
    from core.face_alignment import Alignment
    os.chdir('../')
    align = Alignment()
    a1 = cv2.imread('./images/1.jpg')
    a2 = cv2.imread('./images/2.jpg')
    n1 = cv2.imread('./images/3.png')
    n2 = cv2.imread('./images/4.png')
    s = cv2.imread('./images/s.jpeg')
    z = cv2.imread('./images/z.jpeg')
    s1 = cv2.imread('./images/s1.jpg')
    z1 = cv2.imread('./images/z1.jpg')

    encoder = Encoder()
    print(1)
    ea1 = encoder.generate_features(align.align_face(a1))
    print(2)
    ea2 = encoder.generate_features(align.align_face(a2))
    print(3)
    en1 = encoder.generate_features(align.align_face(n1))
    print(4)
    en2 = encoder.generate_features(align.align_face(n2))
    print(5)
    es = encoder.generate_features(align.align_face(s))
    print(6)
    ez = encoder.generate_features(align.align_face(z))
    print(7)
    es1 = encoder.generate_features(align.align_face(s1))
    print(8)
    ez1 = encoder.generate_features(align.align_face(z))

    ds1 = encoder.distance(ea1, ea2)
    ds2 = encoder.distance(en1, en2)
    dn1 = encoder.distance(ea1, ez)

    dn2 = encoder.distance(ea2, ez)

    dn3 = encoder.distance(es, ez)

    ds3 = encoder.distance(es, es1)
    ds4 = encoder.distance(ez, ez1)
    dn4 = encoder.distance(es, ez1)
    # facenet.pb 1.1739014 1.8518097 1.9747759 1.0059586
    # f2.pb 0.20989718 0.7416147 1.9053222 1.7969146 0.91570365
    # f3.pb 0.20603313 0.7747024 2.061185 1.9731356 0.9667357
    # f4.pb 0.21215439 0.64486355 2.0271614 1.8071315 0.92480683
    # f5.pb 0.26717526 0.6539026 2.0113552 1.7940606 0.93446374
    print(ds1, ds2, ds3, ds4, dn1, dn2, dn3, dn4)
