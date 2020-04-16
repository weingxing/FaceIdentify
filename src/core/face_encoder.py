# 获取人脸特征向量
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import math

from core.load_model import load_model

# 模型路径
# facenet_model_checkpoint = os.path.abspath("./core/models/facenet.pb")
facenet_model_checkpoint = os.path.abspath("./core/models/facenet/facenet.pb")
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

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        # y = self.sess.run(tf.image.per_image_standardization(x))
        return y

    def generate_embedding(self, image):
        # 取得输入层和输出层的张量及训练标志位
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = self.prewhiten(image)
        # 计算特征量
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def distance(self, emb1, emb2):
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff), 1)
        dist = np.sum(np.square(emb1 - emb2))
        return dist


if __name__ == '__main__':
    img1 = cv2.imread('../image/1.jpg')
    img2 = cv2.imread('../image/2.jpg')
    img3 = cv2.imread('../image/3.jpg')
    encoder = Encoder()
    emb1 = encoder.generate_embedding(img1)
    emb2 = encoder.generate_embedding(img2)
    emb3 = encoder.generate_embedding(img3)
    d1 = encoder.distance(emb1, emb2)
    d2 = encoder.distance(emb1, emb3)
    d3 = encoder.distance(emb2, emb3)
    print(d1, d2, d3)
