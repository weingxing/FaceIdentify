import cv2
import matplotlib.pyplot as plt
from face_db import FaceDB
from core.face_recognition import Recognition
from core.face_alignment import Alignment

DB = FaceDB()
recognition = Recognition()

if __name__ == '__main__':
    dic = {
        '邓超': 'C:/Users/24518/Desktop/1/dengchao.jpg',
        '李诞': 'C:/Users/24518/Desktop/1/lidan.jpeg',
        '刘涛': 'C:/Users/24518/Desktop/1/liutao.jpg',
        '宋慧乔': 'C:/Users/24518/Desktop/1/songhuiqiao.jpg',
        '杨幂': 'C:/Users/24518/Desktop/1/yangmi.jpg',
        '杨颖': 'C:/Users/24518/Desktop/1/yangying.jpg',
        '杨紫': 'C:/Users/24518/Desktop/1/yangzi.jpg',
        '岳云鹏': 'C:/Users/24518/Desktop/1/yueyunpeng.jpg',
        '张一山': 'C:/Users/24518/Desktop/1/zhangyishan.png',
        '张雨绮': 'C:/Users/24518/Desktop/1/zhangyuqi.jpg',
        '赵丽颖': 'C:/Users/24518/Desktop/1/zhaoliying.jpg'
    }
    for key in dic.keys():
        alignment = Alignment()
        # print(dic[key])
        image = cv2.imread(dic[key])
        # print(image)
        image = image[:, :, ::-1]
        image = alignment.align_face(image)
        plt.imshow(image)
        plt.show()
        # recognition.add_to_db(key, image)
