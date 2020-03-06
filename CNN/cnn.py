import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist/data/', one_hot=True) # Dataset module

### Constructing Model ########################################################
tfp_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
tfp_output = tf.placeholder(tf.float32, [None, 10])
tfp_keep_prob = tf.placeholder(tf.float32)

"""
    * Convolution Layer Properties
    W1: filter with shape [3, 3, 1, 32] 
        -> [3, 3]: Kernel size
        -> [1]: channel of the input
        -> [32]: nb of filters (nb of output channel)
    L1: execute convolution and it produces [None, 28, 28, 32]
        -> strides[0, 3] are 1s and strides[1, 2] are usually same
        -> padding='SAME': input size and output size are same -> [28, 28]
    max_pool: execute max_pooling and it produces [None, None, 14, 14, 32]
        -> ksize=[1, 2, 2, 1]: filter size is [2, 2] and it chooses one maximum output
"""
# Conv1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
L1 = tf.nn.conv2d(tfp_input, W1, strides=[1, 1, 1, 1], padding='SAME') #[None, 28, 28, 32]
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #[None, 14, 14, 32]
# L1 = tf.nn.dropout(L1, tfp_keep_prob)

# Conv2
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') #[None, 14, 14, 64]
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #[None, 7, 7, 64]
# L2 = tf.nn.dropout(L2, tfp_keep_prob)

# FC: It transforms the multi-dimensional output of convolution layers into 1 dimension.
# The input would be L2 and L2 should be reshaped as 1 dimension.
W3 = tf.Variable(tf.random_normal([7*7*64, 256]))
L3 = tf.reshape(L2, [-1, 7*7*64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, tfp_keep_prob)

# FC: It determines the last output estimating the prediction.
W4 = tf.Variable(tf.random_normal([256, 10]))
pred = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=tfp_output))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

### Trainging ###################################################################
total_epoch = 15
keep_prob = 0.7

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for batch in range(total_batch):
        batch_input, batch_output = mnist.train.next_batch(batch_size)
        batch_input = batch_input.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                                {tfp_input: batch_input,
                                 tfp_output: batch_output,
                                 tfp_keep_prob: keep_prob})

        total_cost += cost_val
        
        print('\rEpoch: {}/{}  >  cost: {}  >  avg. cost: {:.3f}'.format(
                epoch, total_epoch, cost_val, cost_val/total_batch), end='')

    print('\n')

print("training is finished.....")

### Check the accuracy #########################################################
is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(tfp_output, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        {tfp_input: mnist.test.images.reshape(-1, 28, 28, 1),
                         tfp_output: mnist.test.labels,
                         keep_prob: 1}))
