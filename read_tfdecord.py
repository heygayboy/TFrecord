# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


#Use queue to read data
def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'data_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    data = tf.decode_raw(features['data_raw'], tf.int32)
    data = tf.reshape(data,[600]) ##reshape according to the data
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label,[10])
    data = tf.cast(data, tf.float32)
    input_queue = tf.train.slice_input_producer([data, label], shuffle=False)
    # data_batch,label_batch = tf.train.batch([data,label],   #使用shuffle_batch可以随机打乱输入
    #                                         batch_size=batch_size,
    #                                         num_threads=1,   ##too many threads will bring error
    #                                         capacity = 50000) ##capacity is max of queue

    return data,label ##reshape necessar

#test
save_name = 'test'
save_dir = 'data/cnews'
filename = os.path.join(save_dir,save_name+'.tfrecords')
print(filename)
# cwd='data/cnews/train.tfrecorders'
init = tf.initialize_all_variables()
for i in range(10):
    data, label = read_and_decode(r'data/cnews/train.tfrecords')
print(data, label)

#
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(3):
#         train_data, train_label= sess.run([data_batch, label_batch])
#         print(train_data.shape)





