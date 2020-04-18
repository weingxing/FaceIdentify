import json
import requests
import time
import random


f = open('star_name.txt', 'w', encoding='utf-8')


# 获取明星的名字并保存到文件中
def get_page(pages, star_name):
    params = []
    # 请求头，分页数和明星所在的地区
    for i in range(0, 12 * pages + 12, 12):
        params.append({
            'resource_id': 28266,
            'from_mid': 1,
            'format': 'json',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'query': '明星',
            'sort_key': '',
            'sort_type': 1,
            'stat0': '',
            'stat1': star_name,
            'stat2': '',
            'stat3': '',
            'pn': i,
            'rn': 12})

    url = 'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php'
    x = 0
    for param in params:
        try:
            res = requests.get(url, params=param, timeout=50)
            js = json.loads(res.text)
            print(js)
            results = js.get('data')[0].get('result')
        except AttributeError as e:
            print('【错误】出现错误：%s' % e)
            continue
        # 从json中提取明星的名字
        for result in results:
            img_name = result['ename']
            f.write(img_name + '\n',)
        if x % 10 == 0:
            print('第%d页......' % x)
        time.sleep(random.randint(1, 3))
        x += 1


# 从百度上获取明星名字
def get_star_name():
    names = ['内地', '香港', '台湾']
    # 每个地区获取的页数
    sums = [600, 200, 200]
    for i in range(len(names)):
        get_page(sums[i], names[i])

    f.close()


# 删除可能存在错误的名字，存在错误删除问题
def delete_some_name():
    name_set = set()
    with open('star_name.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 5:
                continue
            name_set.add(line)

    print('筛选后的总人数为：%d' % len(name_set))
    with open('star_name.txt', 'w', encoding='utf-8') as f:
        f.writelines(name_set)


if __name__ == '__main__':
    get_star_name()
    delete_some_name()
