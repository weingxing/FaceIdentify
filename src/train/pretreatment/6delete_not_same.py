import requests
import base64
import os
from shutil import move
from random import shuffle
import time
from pretreatment.compress_image import compress_image


def get_token():
    token_url = 'https://aip.baidubce.com/oauth/2.0/token'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&' \
           'client_id=qO4NWXyefbwRwpGxXTjQKpUo&client_secret=jiDMug4L364DVs03LIGoX0TFqZ7TWkWa'
    response = requests.get(host)
    if response:
        return dict(response.json())['access_token']


token = get_token()


def write_log(msg):
    with open('../../err.txt', 'a') as f:
        f.write(msg)


def get_score(image1, image2):
    try:
        url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
        params = "[{\"image\": \"" + image1 + "\", \"image_type\": \"BASE64\"},\
        {\"image\": \"" + image2 + "\", \"image_type\": \"BASE64\"}]"

        access_token = token
        url = url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        response = requests.post(url, data=params, headers=headers)
        if response:
            return dict(response.json())
    except:
        raise AssertionError('访问出错')


def get_image_b64(image_path):
    f = open(image_path, 'rb')
    base64_data = base64.b64encode(f.read())
    f.close()
    return base64_data.decode()


def start():
    # 图片路径
    path = 'F:\\star_image_processed'
    # 取得所有子文件夹名
    dirs = os.listdir(path)
    if len(dirs) == 0:
        raise IOError('文件夹为空')

    for dir in dirs:
        # 子文件夹完整路径
        dir_path = os.path.join(path, dir)
        # 文件夹下图片名
        images = os.listdir(dir_path)
        if len(images) == 0:
            os.removedirs(dir_path)
            continue
        # 随机打乱数据顺序
        shuffle(images)
        # 选取一张图片作为参照
        base_path = os.path.join(dir_path, images[0])
        base = get_image_b64(base_path)
        # 存储清洗后的数据的路径
        cleaned_path = os.path.join('F://cleaned', dir)
        # 参照图片归位
        if not os.path.exists(cleaned_path):
            os.mkdir(cleaned_path)
            move(base_path, cleaned_path)
        else:
            move(base_path, cleaned_path)
        # 开始遍历比对
        for image in images[1:]:
            image_path = os.path.join(dir_path, image)
            print(image_path)
            check = get_image_b64(image_path)
            # 免费接口QPS为2，限制速度，确保请求成功
            time.sleep(1)
            result = get_score(base, check)
            try:
                # 同一个人的概率大于70%，归位
                if result['result']['score'] > 70:
                    move(image_path, cleaned_path)
                else:
                    print('移除，%s' % image_path)
                    new_path = os.path.join('F:\\error', dir)
                    if not os.path.exists(new_path):
                        os.mkdir(new_path)
                    write_log('\n相似度低：\n' + str(result) + '\n' + new_path)
                    move(image_path, new_path)
            except (KeyError, TypeError):
                try:
                    # 出错可能：图片太大  压缩图片
                    new_image = compress_image(infile=image_path)
                    # 删除原图片
                    os.remove(image_path)
                except ValueError:
                    raise ValueError
                try:
                    if get_score(base, get_image_b64(new_image))['result']['score'] > 70:
                        write_log('\n出现错误，尝试处理成功\n' + str(result) + '\n\n')
                        print('保留：%s' % new_image)
                        move(new_image, cleaned_path)
                except:
                    new_path = os.path.join('F:\\fail', dir)
                    if not os.path.exists(new_path):
                        os.mkdir(new_path)
                    move(new_image, new_path)
                    write_log('\n出现错误，尝试处理失败\n' + str(result) +
                              '\n图片路径：' + new_path + '\n\n')


while True:
    try:
        start()
    except AssertionError as e:
        print(e)
        write_log('网络请求错误:' + str(e))
        time.sleep(60)
    except FileNotFoundError as e:
        write_log('Error:' + str(e))
    except IOError as e:
        print(e)
        break
