import os
import shutil
import cv2
import gc
import numpy as np
from centerface import CenterFace


# 删除两个人脸以上的图片或者没有人脸的图片
def delete_image(image_path):
    centerface = CenterFace(landmarks=True)
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        h, w = img.shape[:2]
        result, lms = centerface(img, h, w, threshold=0.35)
        print(len(result))
        if len(result) != 1:
            print('{} 有{} 个人，删除...'.format(image_path, len(result)))
            os.remove(image_path)
    except Exception as e:
        print(e)
    finally:
        del centerface
        gc.collect()


if __name__ == '__main__':
    father_path = 'star_image'
    processed_path = 'star_image_processed'

    name_paths = os.listdir(father_path)
    for name_path in name_paths:
        print('{}正在检测 {} 图片...'.format(name_paths.index(name_path), name_path))
        image_paths = os.listdir(os.path.join(father_path, name_path))
        for image_path in image_paths:
            # 获取图片路径
            img_path = os.path.join(father_path, name_path, image_path)
            delete_image(img_path)
        shutil.move(src=os.path.join(father_path, name_path), dst=os.path.join(processed_path, name_path))

    print('处理完成')

