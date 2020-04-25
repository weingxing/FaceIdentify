import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from core.face_encoder import Encoder
from core.face_alignment import Alignment

encoder = Encoder()
# 使图片正常显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# 图片封装类
class Image:
    def __init__(self, base, name, file):
        # 数据集根目录
        self.base = base
        # 目录名
        self.name = name
        # 图像文件名
        self.file = file

    def __repr__(self):
        return self.image_path()

    #   取得图像路径
    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


# 载入数据
def load_data(path):
    data = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # 检查文件名后缀
            ext = os.path.splitext(f)[1]
            exts = ['.jpg', 'jpeg', '.png']
            if ext in exts:
                data.append(Image(path, i, f))
    return np.array(data)


# 训练分类器(选用 KNN 和 SVM)
def train(X_train, y_train, X_test, y_test, save_path):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()

    # 训练 knn 和 svm
    print('训练KNN中...')
    knn.fit(X_train, y_train)
    print('训练SVM中...')
    svc.fit(X_train, y_train)

    # 计算分类器精度
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
    save_models(knn, 'knn.pkl', save_path)
    save_models(svc, 'svc.pkl', save_path)
    return knn, svc


# 保存训练后的分类器
def save_models(model, name, path):
    if not os.path.exists(path):
        os.mkdir(path)
    print('保存%s' % name)
    f = open(os.path.join(path, name), 'wb')
    pickle.dump(model, f)
    f.close()


# 载入分类器
def load_model(path):
    print('载入分类器：%s' % path)
    f = open(path, 'rb')
    model = pickle.load(f)
    return model


# 取得图片特征向量
def get_features(data):
    # 初始化特征向量矩阵，大小： 待分类人数 x 512
    features = np.zeros((data.shape[0], 512))
    for i, m in enumerate(data):
        align = Alignment()
        print('导出 %s 的特征向量...' % m)
        image = cv2.imread(str(m))
        aligned_image = align.align_face(image)
        features[i] = encoder.generate_features(aligned_image)
    return features


# 分割数据为 训练集 和 测试集
def split_data(data, save_path):
    features = get_features(data)
    # 取得数据标签（人名）
    labels = np.array([i.name for i in data])
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    # 数字化标签
    y = label_encoder.transform(labels)
    # 1/3 测试， 2/3 训练
    train_idx = np.arange(labels.shape[0]) % 3 != 0
    test_idx = np.arange(labels.shape[0]) % 3 == 0

    # 训练数据
    X_train = features[train_idx]
    # 测试数据
    X_test = features[test_idx]
    # 训练数据标签
    y_train = y[train_idx]
    # 测试数据标签
    y_test = y[test_idx]
    # 保存标签编码器
    save_models(label_encoder, 'encoder.pkl', save_path)
    return [X_train, y_train], [X_test, y_test]


def start(args):
    data = load_data(args['data'])

    if args['flag'] == 'train':
        train_data, test_data = split_data(data, args['path'])
        knn, svc = train(train_data[0], train_data[1], test_data[0], test_data[1], args['path'])
    else:
        knn = load_model(os.path.join(args['path'], 'knn.pkl'))
        svc = load_model(os.path.join(args['path'], 'svc.pkl'))
        label_encoder = load_model(os.path.join(args['path'], 'encoder.pkl'))

        images = os.listdir(args['image_path'])
        for i in images:
            image = os.path.join(args['image_path'], i)
            image = cv2.imread(image)
            feature = encoder.generate_features(Alignment().align_face(image))
            prediction = svc.predict([feature])
            identity = label_encoder.inverse_transform(prediction)[0]
            print(identity, max((svc._predict_proba_lr([feature])[0])))
            plt.imshow(image[:, :, ::-1])
            plt.title(f'识别结果是{identity}')
            plt.show()


if __name__ == '__main__':
    args = {
        # 数据目录
        'data': 'G:\\2',
        # 训练标志位，训练分类器设为 train，使用时随意设置
        'flag': 'trai',
        # 模型存储路径
        'path': './classify',
        # 测试图片
        'image_path': 'G:\\1'
    }

    try:
        start(args)
    except Exception as e:
        print(e)
