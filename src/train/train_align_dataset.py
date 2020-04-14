# 对图像进行裁剪，将图片转化为 只包含人脸的 160*160 大小的图像
import os
import numpy as np
import core
import random
from time import sleep
from core.centerface import CenterFace
import cv2
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank

REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
], np.float32)


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args['output_dir'])
    # 如果不存在输出文件夹，新建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = core.get_dataset(args['input_dir'])
    images_total = 0
    successfully_aligned = 0
    # 打乱数据
    # if args['random_order']:
    #     random.shuffle(dataset)
    # 遍历处理数据
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        # 如果不存在输出文件夹，新建
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        # 打乱数据
        # if args['random_order']:
        #     random.shuffle(cls.image_paths)
        # 遍历处理同一个人的数据
        for image_path in cls.image_paths:
            images_total += 1
            # 取得文件路径
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print('路径：' + image_path)
            if not os.path.exists(output_filename):
                try:
                    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                except (IOError, ValueError) as e:
                    error = '{}: {}'.format(image_path, e)
                    print(error)
                else:
                    filename_base, file_extension = os.path.splitext(output_filename)
                    output_filename_n = "{}{}".format(filename_base, file_extension)

                    centerface = CenterFace(landmarks=True)
                    h, w = img.shape[:2]
                    boxes, lms = centerface(img, h, w, threshold=0.35)
                    if len(lms) > 0:
                        landmarks = lms[0]
                        tmp = []
                        i = 0
                        while i < len(landmarks):
                            tmp.append([landmarks[i], landmarks[i + 1]])
                            i = i + 2

                        # 眼睛、鼻子、嘴巴 共五个关键点
                        source_point = np.array(tmp, np.float32)
                        similar_trans_matrix = get_dist_point(source_point,
                                                          REFERENCE_FACIAL_POINTS)
                        # 仿射变换
                        aligned_face = cv2.warpAffine(img, similar_trans_matrix, (112, 112))
                        # 调整大小
                        aligned_face = cv2.resize(aligned_face, (args['image_size'], args['image_size']))
                        cv2.imencode('.png', aligned_face)[1].tofile(output_filename_n)

                        successfully_aligned += 1
                        print('目前成功处理数量: %d' % successfully_aligned)

    print('处理完成，总图片数量: %d' % images_total)
    print('成功处理数量: %d' % successfully_aligned)


def get_dist_point(uv, xy, K=2):
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

    tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])
    T = inv(tinv)
    T[:, 2] = np.array([0, 0, 1])
    T = T[:, 0:2].T
    return T


if __name__ == '__main__':
    args = {
        # 输入路径
        'input_dir': 'G:\\lfw',
        # 输出路径
        'output_dir': 'G:\\lfw_160',
        # 图片大小
        'image_size': 160,
        # 随机选取
        'random_order': False
    }

    main(args)
