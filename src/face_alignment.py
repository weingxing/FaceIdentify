# 人脸检测、对齐
import cv2
import numpy as np
import matplotlib.pyplot as plt

from centerface import CenterFace


class Alignment:
    def __init__(self):
        self.centerface = CenterFace(landmarks=True)

    def align_face(self, img):
        dets, key_point = self.detect(img)
        key_point = key_point[0]

        # 根据两个鼻子和眼睛进行3点对齐
        eye1 = key_point[:2]
        eye2 = key_point[2:4]
        noise = key_point[4:6]
        source_point = np.array([eye1, eye2, noise], dtype=np.float32)

        eye1_normal = [int(x) for x in "81, 167".split(',')]
        eye2_normal = [int(x) for x in "159, 167".split(',')]
        noise_normal = [int(x) for x in "120, 215".split(',')]
        # 设置的人脸标准模型
        dst_point = np.array([eye1_normal, eye2_normal, noise_normal], dtype=np.float32)

        transform = cv2.getAffineTransform(source_point, dst_point)

        image_size = tuple([int(x) for x in "266, 298".split(',')])
        img_new = cv2.warpAffine(img, transform, image_size)
        img_new = cv2.resize(img_new, (160, 160))

        return img_new

    def detect(self, img):
        h, w = img.shape[:2]
        dets, lms = self.centerface(img, h, w, threshold=0.35)
        return dets, lms


if __name__ == '__main__':
    pic = './1.jpg'
    centerface = CenterFace(landmarks=True)
    img = cv2.imread(pic)
    # BGR 转 RGB
    img = img[:, :, ::-1]
    h, w = img.shape[:2]
    dets, lms = centerface(img, h, w, threshold=0.35)

    if len(dets):
        align = Alignment()
        aligned = align.align_face(img, lms)
        plt.imshow(aligned)
        plt.show()
    else:
        print('not found face')
