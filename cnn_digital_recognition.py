# coding=utf-8
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_train = True

mnist_data = input_data.read_data_sets('./mnist_data/', one_hot=True)

def init_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, mean=0.0, stddev=0.05))


def cnn_model(x):
    with tf.variable_scope('conv_1'):

        x = tf.reshape(x, [-1, 28, 28, 1])

        conv1_w = init_weights([5, 5, 1, 32])
        conv1_b = init_weights([32])


        x_conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b


        x_relu1 = tf.nn.relu(x_conv1)

        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



    with tf.variable_scope('conv_2'):

        conv2_w = init_weights([5, 5, 32, 64])
        conv2_b = init_weights([64])

        x_conv2 = tf.nn.conv2d(x_pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

        x_relu2 = tf.nn.relu(x_conv2)

        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1 ,2, 2, 1], padding='SAME')


    with tf.variable_scope('full_connection'):

        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        fc_w = init_weights([7 * 7 * 64, 10])
        fc_b = init_weights([10])

        y_predict = tf.matmul(x_fc, fc_w) + fc_b

    return y_predict


def mnist_demo():
    with tf.variable_scope("original_data"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        y_predict = cnn_model(x)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_predict))

    with tf.variable_scope('optimzier'):

        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.variable_scope("accuracy"):
        temp_acc = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(temp_acc, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as  sess:
        sess.run(tf.global_variables_initializer())

        if is_train:
            for i in range(1000):
                x_i, y_i = mnist_data.train.next_batch(50)
                sess.run(train_op, feed_dict={x: x_i, y_true: y_i})

                loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict={x: x_i, y_true: y_i})

                print('第%d次, loss=%f, accuracy=%f' % (i + 1, loss_i, accuracy_i))
                saver.save(sess, './accuracy_save/')

        else:
            if os.path.exists('./accuracy_save/checkpoint'):
                saver.restore(sess, './forecast_save')

            for i in range(100):
                x_i, y_i = mnist_data.test.next_batch(1)
                result_true = sess.run(y_predict, feed_dict={x: x_i, y_true: y_i})
                result_predict = sess.run(y_predict, feed_dict={x: x_i, y_true: y_i})

                true_num = tf.argmax(result_true, 1).eval()
                predict_num = tf.argmax(result_predict, 1).eval()

                print('第%d个样本，真实值=%d, 预测值=%d' % (i + 1, true_num, predict_num))


if __name__ == '__main__':
    mnist_demo()