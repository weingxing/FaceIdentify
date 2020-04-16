import imghdr
import os
import numpy
from PIL import Image


# 删除不是JPEG或者JPG格式的图片
def delete_error_image(father_path):
    try:
        image_dirs = os.listdir(father_path)
        for image_dir in image_dirs:
            image_dir = os.path.join(father_path, image_dir)
            if os.path.isdir(image_dir):
                images = os.listdir(image_dir)
                for image in images:
                    image = os.path.join(image_dir, image)
                    try:
                        image_type = imghdr.what(image)
                        # 如果图片格式不是JPEG同时也不是JPG就删除图片
                        if image_type is not 'jpeg' and image_type is not 'jpg':
                            os.remove(image)
                            print('已删除：%s' % image)
                            continue
                        # 删除灰度图
                        img = numpy.array(Image.open(image))
                        if len(img.shape) is 2:
                            os.remove(image)
                            print('已删除：%s' % image)
                    except:
                        os.remove(image)
                        print('已删除：%s' % image)
    except:
        pass


if __name__ == '__main__':
    print('开始删除错误图片')
    delete_error_image('star_image/')