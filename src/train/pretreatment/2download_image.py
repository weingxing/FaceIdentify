import os
import re
import uuid
import time
import random
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 获取百度图片下载图片
def download_image(key_word, download_max):
    download_sum = 0
    str_gsm = '80'
    # 把每个明显的图片存放在单独一个文件夹中
    save_path = 'star_image' + '/' + key_word
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    while download_sum < download_max:
        # 下载次数超过指定值就停止下载
        if download_sum >= download_max:
            break
        str_pn = str(download_sum)
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
              'word=' + key_word + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'
        print('正在下载 %s 的第 %d 张图片.....' % (key_word, download_sum))
        try:
            result = requests.get(url, timeout=30).text
            img_urls = re.findall('"objURL":"(.*?)",', result, re.S)
            if len(img_urls) < 1:
                break
            for img_url in img_urls:
                img = requests.get(img_url, timeout=30)
                img_name = save_path + '/' + str(uuid.uuid1()) + '.jpg'
                # 保存图片
                with open(img_name, 'wb') as f:
                    f.write(img.content)
                with open('image_url_list.txt', 'a+', encoding='utf-8') as f:
                    f.write(img_name + '\t' + img_url + '\n')
                download_sum += 1
                if download_sum >= download_max:
                    break
        except Exception as e:
            print('【错误】当前图片无法下载，%s' % e)
            download_sum += 1
            continue
    print('下载完成')


if __name__ == '__main__':
    # 图片链接文档
    with open('image_url_list.txt', 'w', encoding='utf-8') as f_u:
        pass

    max_sum = 10
    with open('star_name.txt', 'r', encoding='utf-8') as f:
        key_words = f.readlines()
        
    i = 0
    for key_word in key_words:
        key_word = key_word.replace('\n', '')
        download_image(key_word, max_sum)
        t = random.randint(10, 90)
        print(i)
        if i == 10:
            print('随机睡眠 %d 秒' % t)
            time.sleep(t)
            i = 0
        i = i + 1
    print('\n全部图片下载完成\n')

    # 爬取完成邮件通知
    msg_from = 'oxygen@mapletown.xyz'
    passwd = '这是密码'
    msg_to = '2451809588@qq.com'

    subject = "数据爬取完成"
    content = "数据爬取完成, 可以下载了"
    
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = msg_from

    try:
        s = smtplib.SMTP_SSL("smtp.exmail.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print('邮件发送成功')
    except s.SMTPException as e:
        print(e)
    finally:
        s.quit()
