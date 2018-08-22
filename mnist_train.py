import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "path/to/model"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
                       name="x-input")

    y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, 0, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 每次选取batch_size个样本进行训练
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            # 通过选取样本训练神经网络并更新参数
            #sess.run(train_step, feed_dict={x: trainx[start:end], y_: trainy[start:end]})
            # 每迭代1000次输出一次日志信息
            if i % 1000 == 0:
                # 计算所有数据的交叉熵
                #total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                #total_mse = sess.run(loss, feed_dict={x: trainx, y_: trainy})
                #train_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y})
                # 输出交叉熵之和
                #print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
                print("After %d training step(s), loss on training "
                      "batch is %g" % (i, loss_value))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("save the model")


def main(argv=None):
    mnist = input_data.read_data_sets("path/to/mnist_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
