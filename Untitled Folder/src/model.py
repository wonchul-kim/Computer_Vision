import tensorflow as tf

def weights(shape, stddev=0.01):
    return tf.get_variable("weights", shape, tf.float32, \
                        tf.random_normal_initializer(mean=0.0, stddev=stddev))

def biases(shape, )