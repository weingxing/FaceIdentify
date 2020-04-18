from tensorflow.python.framework import graph_util
import tensorflow as tf
import os
import sys
from six.moves import xrange


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args['model_dir'])
            meta_file, ckpt_file = get_model_filenames(os.path.expanduser(args['model_dir']))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            model_dir_exp = os.path.expanduser(args['model_dir'])
            saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file), clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
            
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'embeddings,label_batch')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args['output_file'], 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args['output_file']))


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
 
    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
                node.name.startswith('image_batch') or node.name.startswith('label_batch') or
                node.name.startswith('phase_train') or node.name.startswith('Logits')):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","),
        variable_names_whitelist=whitelist_names)
    return output_graph_def


if __name__ == '__main__':
    args = {
        'model_dir': 'G:/1',
        'output_file': 'G:/facenet.pb'
    }
    main(args)
