import tensorflow as tf

def weights(shape, stddev=0.01):
    return tf.get_variable("weights", shape, tf.float32, \
                        tf.random_normal_initializer(mean=0.0, stddev=stddev))

def biases(shape, value=1.0):
    return tf.get_variable('biases', shape, tf.float32, \
                        tf.constant_initializer(value=vlaue))

def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pooling(x, width, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, width, width, 1], \
                strides=[1, stride, stride, 1], padding=padding)

def add_conv2d(x, filter_width, stride, nb_output_channel, padding='SAME'):
    nb_intput_channel = int(x.get_shape()[-1])
    w = weights([filter_width, filter_width, nb_input_channel, nb_output_channel], stddev=0.01)
    b = biases([nb_output_channel], value=0.1)

    return conv2d(x, w, stride, padding=padding) + b

def add_fc(x, output_)
