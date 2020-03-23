# 人脸检测、对齐
import cv2
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import matplotlib.pyplot as plt

from centerface import CenterFace

# 压制警告
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class Alignment:
    def __init__(self):
        self.centerface = CenterFace(landmarks=True)
        self.REFERENCE_FACIAL_POINTS = np.array([
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ], np.float32)

    def align_face(self, img):
        box, landmarks = self.detect(img)
        landmarks = landmarks[0]
        tmp = []
        i = 0
        while i < len(landmarks):
            tmp.append([landmarks[i], landmarks[i + 1]])
            i = i + 2

        # 眼睛、鼻子、嘴巴 共五个关键点
        source_point = np.array(tmp, np.float32)
        similar_trans_matrix = self.get_dist_point(source_point, self.REFERENCE_FACIAL_POINTS)
        aligned_face = cv2.warpAffine(img, similar_trans_matrix, (112, 112))
        aligned_face = cv2.resize(aligned_face, (160, 160))

        return aligned_face

    def detect(self, img):
        h, w = img.shape[:2]
        dets, lms = self.centerface(img, h, w, threshold=0.35)
        return dets, lms

    def get_dist_point(self, uv, xy, K=2):
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))
        y = xy[:, 1].reshape((-1, 1))

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))

        u = uv[:, 0].reshape((-1, 1))
        v = uv[:, 1].reshape((-1, 1))
        U = np.vstack((u, v))

        # X * r = U
        if rank(X) >= 2 * K:
            r, a, b, c = lstsq(X, U)
            r = np.squeeze(r)
        else:
            raise Exception('cp2tform:twoUniquePointsReq')

        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ])
        T = inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])
        T = T[:, 0:2].T
        return T


if __name__ == '__main__':
    pic = './imgs/2.jpg'
    centerface = CenterFace(landmarks=True)
    img = cv2.imread(pic)
    # BGR 转 RGB
    img = img[:, :, ::-1]
    h, w = img.shape[:2]
    dets, lms = centerface(img, h, w, threshold=0.35)

    if len(dets):
        align = Alignment()
        aligned = align.align_face(img)
        plt.imshow(aligned)
        plt.show()
    else:
        print('not found face')
