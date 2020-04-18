from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import facenet
import lfw

from tensorflow.python.ops import data_flow_ops
from six.moves import xrange


def main(args):
    # 导入网络架构模型
    network = importlib.import_module(args['model_def'])
    # 用当前日期来命名模型
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    # 日志保存目录
    log_dir = os.path.join(os.path.expanduser(args['logs_base_dir']), subdir)
    # 没有日志文件就创建一个
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args['models_base_dir']), subdir)
    # 没有模型保存目录就创建一个
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 保存参数日志
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # 设置随机数种子
    np.random.seed(seed=args['seed'])

    # 获取数据集，train_set是包含文件路径与标签的集合
    # 包含图片地址的（image_paths）以及对应的人名(name)
    train_set = facenet.get_dataset(args['data_dir'])

    print('模型目录: %s' % model_dir)
    print('log目录: %s' % log_dir)
    # 判断是否有预训练模型
    if args['pretrained_model']:
        print('Pre-trained model: %s' % os.path.expanduser(args['pretrained_model']))

    if args['lfw_dir']:
        print('LFW目录: %s' % args['lfw_dir'])
        # 读取用于测试的pairs文件
        pairs = lfw.read_pairs(os.path.expanduser(args['lfw_pairs']))
        # 获取对应的路径
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args['lfw_dir']),
                                                 pairs, args['lfw_file_ext'])

    # 建立图
    with tf.Graph().as_default():
        tf.set_random_seed(args['seed'])
        global_step = tf.Variable(0, trainable=False)
        # 学习率
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        # 批大小
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        # 用于判断是训练还是测试
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # 图像路径
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        # 图像标签
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')
        # 新建一个队列，数据流操作，先入先出
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        preprocess_threads = 4
        images_and_labels = []
        for _ in range(preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                # 随机水平反转
                if args['random_flip']:
                    image = tf.image.random_flip_left_right(image)

                image.set_shape((args['image_size'], args['image_size'], 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args['image_size'], args['image_size'], 3), ()], enqueue_many=True,
            capacity=4 * preprocess_threads * args['batch_size'],
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # 构造计算图
        # 其中prelogits是最后一层的输出
        prelogits, _ = network.inference(image_batch, args['keep_probability'],
                                         phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=args['embedding_size'],
                                         weight_decay=args['weight_decay'])

        # L2正则化
        # embeddings = tf.nn.l2_normalize
        # 输入向量, L2范化的维数（取0（列L2范化）或1（行L2范化））
        # 泛化的最小值边界, name='embeddings')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # 计算 triplet_loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [
            -1, 3, args['embedding_size']]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args['alpha'])

        # 将指数衰减应用在学习率上
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args['learning_rate_decay_epochs']\
                                                   * args['epoch_size'],
                                                   args['learning_rate_decay_factor'],
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # 计算损失
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # 构建L2正则化
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # 确定优化方法并根据损失函求梯度，每更新一次参数，global_step 会加 1
        train_op = facenet.train(total_loss, global_step, args['optimizer'],
                                 learning_rate, args['moving_average_decay'],
                                 tf.global_variables())

        # 创建一个saver用来保存或者从内存中读取一个模型参数
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        summary_op = tf.summary.merge_all()

        # 设置显存比例
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args['gpu_memory_fraction'])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # 初始化变量
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        # 写log文件
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # 获取线程
        coord = tf.train.Coordinator()
        # 将队列中的多用sunner开始执行
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # 读入预训练模型（如果有）
            if args['pretrained_model']:
                print('载入预训练模型: %s' % args['pretrained_model'])
                # saver.restore(sess, os.path.expanduser(args['pretrained_model']))
                facenet.load_model(args['pretrained_model'])

            epoch = 0
            # 将所有数据过一遍的次数
            while epoch < args['max_nrof_epochs']:
                step = sess.run(global_step, feed_dict=None)
                # epoch_size是一个epoch中批的个数
                # epoch是全局的批处理个数以一个epoch中。。。这个epoch将用于求学习率
                epoch = step // args['epoch_size']
                # 训练一个epoch
                train(args, sess, train_set, epoch, image_paths_placeholder,
                      labels_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder,
                      phase_train_placeholder, enqueue_op,
                      input_queue, global_step,
                      embeddings, total_loss, train_op, summary_op,
                      summary_writer, args['learning_rate_schedule_file'],
                      args['embedding_size'], anchor, positive, negative, triplet_loss)

                # 保存变量和metagraph（如果不存在）
                save_variables_and_metagraph(sess, saver, summary_writer,
                                             model_dir, subdir, step)

                # 使用lfw评价当前模型
                if args['lfw_dir']:
                    evaluate(sess, lfw_paths, embeddings, labels_batch,
                             image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder,
                             phase_train_placeholder, enqueue_op,
                             actual_issame, args['batch_size'],
                             args['lfw_nrof_folds'], log_dir, step, summary_writer,
                             args['embedding_size'])

    return model_dir


# 训练
def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0
    # 学习率大于 0
    if args['learning_rate'] > 0.0:
        lr = args['learning_rate']
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    # 运行一个批处理
    while batch_number < args['epoch_size']:
        # 随机选取数据
        # 1800 个
        image_paths, num_per_class = sample_people(dataset,
                                                   args['people_per_batch'],
                                                   args['images_per_person'])

        print('进行前向传播:', end='')
        start_time = time.time()
        examples = args['people_per_batch'] * args['images_per_person']
        # 将输入的1800个图像变成了600*3的二维数据，
        # reshape(a,(-1,3))表示只给定列数为3, 行数自行算出
        labels_array = np.reshape(np.arange(examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        # 开辟一个新的线程用于在内存里读取数据
        sess.run(enqueue_op,
                 {image_paths_placeholder: image_paths_array,
                  labels_placeholder: labels_array})
        emb_array = np.zeros((examples, embedding_size))
        # nrof_batches，1800/90=20
        nrof_batches = int(np.ceil(examples / args['batch_size']))
        # 批处理取得人脸特征，默认为20个批
        for i in range(nrof_batches):
            batch_size = min(examples - i * args['batch_size'], args['batch_size'])
            emb, lab = sess.run([embeddings, labels_batch],
                                feed_dict={batch_size_placeholder: batch_size,
                                           learning_rate_placeholder: lr,
                                           phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        print('选择合适的三元组')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array,
                                                                    num_per_class,
                                                                    image_paths,
                                                                    args['people_per_batch'],
                                                                    args['alpha'])
        selection_time = time.time() - start_time
        print('(random_negs, triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # 使用选定的三元组训练
        nrof_batches = int(np.ceil(nrof_triplets * 3 / args['batch_size']))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        # 读取数据的操作
        sess.run(enqueue_op,
                 {image_paths_placeholder: triplet_paths_array,
                  labels_placeholder: labels_array})
        examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        # 根据求出的特征计算triplet损失函数并进行优化
        summary = tf.Summary()
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(examples - i * args['batch_size'], args['batch_size'])
            feed_dict = {batch_size_placeholder: batch_size,
                         learning_rate_placeholder: lr,
                         phase_train_placeholder: True}
            # sess run 有5个输入，fetches，先运行loss
            # 前向计算的损失，train_op是根据损失来计算梯度，来对参数进行优化
            err, _, step, emb, lab = sess.run([loss, train_op, global_step,
                                               embeddings, labels_batch],
                                              feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, args['epoch_size'], duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        # 将验证损失和准确性添加到summary
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


# 选择一个用于训练的三元组
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    """ 
    VGG Face: Choosing good triplets is crucial and should strike a balance between
    selecting informative (i.e. challenging) examples and swamping training with examples that
    are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    the image n at random, but only between the ones that violate the triplet loss margin. The
    latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    choosing the maximally violating example, as often done in structured output learning.
    
    选择好的三元组是至关重要的，应该选择对于深度学习网络具有挑战的例子。
    """
    # 遍历每一个人
    for i in xrange(people_per_batch):
        # 这个人对应了几张图片
        nrof_images = int(nrof_images_per_class[i])
        # 遍历第i个人的所有图片
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            # For every possible positive pair.
            for pair in xrange(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # FaceNet
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]
                # VGG Face
                # all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx],
                    #    nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


# 从数据集中进行抽样图片，输入参数：
# 1、训练数据集  2、每一个batch抽样多少人 3、每个人抽样多少张
# 默认：选择40张人脸图片作为正样本，随机筛选其他人脸图片作为负样本
def sample_people(dataset, people_per_batch, images_per_person):
    # 总共应该抽取多少张    people_per_batch：45  images_per_person：40
    nrof_images = people_per_batch * images_per_person
    # 数据量不够，暂时用这个代替
    # nrof_images = 900

    # 数据集中一共有多少个不同人的图片
    nrof_classes = len(dataset)

    class_indices = np.arange(nrof_classes)
    # 随机打乱数据
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # 循环抽样，直到满足最小批
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        # nrof_images_from_class = min(nrof_images_in_class, 20, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        # 图片路径 image_paths_for_class：每一类的图片
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        # 图片label（即文件名）
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
    return image_paths, num_per_class


# 评价
def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder,
             labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder,
             phase_train_placeholder, enqueue_op, actual_issame,
             batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()

    print('前向传播计算特征量: ', end='')
    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array,
                          labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch],
                            feed_dict={batch_size_placeholder: batch_size,
                                       learning_rate_placeholder: 0.0,
                                       phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame,
                                                     nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time

    summary = tf.Summary()
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


# 保存模型
def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # 保存checkpoint
    print('保存变量')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('变量保存完成，耗时：%.2f s' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('保存metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('保存完成，耗时：%.2f s' % save_time_metagraph)
    summary = tf.Summary()
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


# 从文件中读取学习率
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


if __name__ == '__main__':
    args = {
        # 日志目录
        'logs_base_dir': './logs/facenet',
        # 模型目录
        'models_base_dir': './models/facenet',
        # GPU显存占用比例
        'gpu_memory_fraction': 0.75,
        # 预训练模型目录
        'pretrained_model': '',
        # 对齐后的数据路径（训练集）
        'data_dir': './data',
        # 网络模型（模块.名称）（models目录下的inception_resnet_v1.py）
        'model_def': 'inception_resnet_v1',
        # epoch
        'max_nrof_epochs': 500,
        # 批大小
        'batch_size': 90,
        # 输入图片大小
        'image_size': 160,
        # 每批的人数
        'people_per_batch': 45,
        # 每个人的图片数
        'images_per_person': 20,
        # 每个epoch的批的数量
        'epoch_size': 1000,
        # 正三元组到负三元组的边距
        'alpha': 0.2,
        # 特征向量维度
        'embedding_size': 512,
        # 对训练图像执行随机水平翻转
        'random_flip': True,
        # 全连接层的保留概率
        'keep_probability': 0.8,
        # L2 正则化
        'weight_decay': 1e-4,
        # 优化算法
        'optimizer': 'ADAM',
        # 学习率
        # 如果设为负值，可以在学习率计划文件中
        # 指定每个epoch的学习率
        'learning_rate': 0.1,
        # 包含将learning_rate设置为负数时使用的学习率计划文件
        'learning_rate_schedule_file': './learning_rates.txt',
        # 学习率衰减之间的epoch数
        'learning_rate_decay_epochs': 100,
        # 学习率衰减因子
        'learning_rate_decay_factor': 1.0,
        # 跟踪训练参数的指数衰减
        'moving_average_decay': 0.9999,
        # 随机数种子
        'seed': 666,

        # lfw数据集的pairs文件路径
        'lfw_pairs': './lfw_160/pairs.txt',
        # lfw数据集的文件后缀（png/jpg)
        'lfw_file_ext': 'png',
        # 对齐后的lfw数据路径
        'lfw_dir': './lfw_160',
        # 交叉验证的文件夹数（主要用于测试）
        'lfw_nrof_folds': 10
    }

    main(args)
