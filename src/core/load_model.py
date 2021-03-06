import os
import tensorflow as tf
import re
from tensorflow.python.platform import gfile


def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    print()
    if os.path.isfile(model_exp):
        print('PB文件：%s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph文件名：%s' % meta_file)
        print('Checkpoint文件名：%s' % ckpt_file)
        print('载入模型中...')

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file),
                                           input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('找不到meta文件，目录：(%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('meta文件多于一个：(%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
