# coding: utf-8

import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from cnn_model import TCNNConfig, TextCNN

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = base_dir

config = TCNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)



def convert_to_tfrecord(data_dir, save_dir, save_name):
    x_data, y_data = process_file(data_dir, word_to_id, cat_to_id, config.seq_length)
    # y_data = np.argmax(y_onehot, 1)

    filename = os.path.join(save_dir,save_name+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')

    for n_sample, label in zip(x_data, y_data):
        data_one_dimension = np.squeeze(n_sample)
        data_raw = data_one_dimension.tobytes()
        label_one_dimension = np.squeeze(label)
        label_raw = label_one_dimension.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print('Transform done!')

convert_to_tfrecord(train_dir, save_dir,'train')
convert_to_tfrecord(val_dir, save_dir,'val')
convert_to_tfrecord(test_dir, save_dir,'test')