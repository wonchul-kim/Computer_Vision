import tensorflow as tf

def weights(shape, stddev=0.01):
    return tf.get_variable("weights", shape, tf.float32, \
                        tf.random_normal_initializer(mean=0.0, stddev=stddev))

def biases(shape, value=1.0):
    return tf.get_variable('biases', shape, tf.float32, \
                        tf.constant_initializer(value=vlaue))

def conv2d(x, W, stride, padding='SAME'):
    '''
        x: tf.Tensor (N, H, W, C)
        W: tf.Tensor (fh, fw, ic, oc)
    '''

    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def max_pooling(x, width, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, width, width, 1], \
                strides=[1, stride, stride, 1], padding=padding)


def conv_layer