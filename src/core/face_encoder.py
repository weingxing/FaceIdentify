# 获取人脸特征向量
import os
import cv2
import time
import numpy as np
import tensorflow as tf

from core.centerface import CenterFace
from core.load_model import load_model


# 模型路径
# facenet_model_checkpoint = os.path.abspath("./core/models/facenet.pb")
facenet_model_checkpoint = os.path.\
    abspath("./core/models/facenet/facenet.pb")
# 预训练模型
# facenet_model_checkpoint = os.path.abspath("G:\\facenet_pretrained\\20180408-102900.pb")


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Detection:
    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def find_faces(self, image):
        # BGR转为RGB
        image = image[:, :, ::-1]
        centerface = CenterFace(landmarks=True)
        h, w = image.shape[:2]
        bounding_boxes, lms = centerface(image, h, w, threshold=0.35)

        faces = []

        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3],
                      face.bounding_box[0]:face.bounding_box[2], :]

            face.image = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_LINEAR)

            faces.append(face)
        return faces[0]


class Encoder:
    def __init__(self):
        self.detection = Detection()
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
        return y

    def generate_embedding(self, image):
        # 取得输入层和输出层的张量
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        face = self.detection.find_faces(image)

        prewhiten_face = self.prewhiten(face.image)
        # 前行传播计算特征量
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def distance(self, emb1, emb2):
        return np.sum(np.square(emb1 - emb2))


if __name__ == '__main__':
    img1 = cv2.imread('../imgs/1.jpg')
    img2 = cv2.imread('../imgs/2.jpg')
    img3 = cv2.imread('../imgs/3.jpg')
    encoder = Encoder()
    emb1 = encoder.generate_embedding(img1)
    emb2 = encoder.generate_embedding(img2)
    emb3 = encoder.generate_embedding(img3)
    d1 = encoder.distance(emb1, emb2)
    d2 = encoder.distance(emb1, emb3)
    d3 = encoder.distance(emb2, emb3)
    print(d1, d2, d3)
