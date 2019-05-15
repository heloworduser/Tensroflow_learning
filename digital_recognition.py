# coding=utf-8
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_train = False

mnist_data = input_data.read_data_sets('./mnist_data/', one_hot=True)


def mnist_demo():
    with tf.variable_scope("original_data"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    with tf.variable_scope("model_param"):
        w = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], mean=0.0, stddev=0.05), name="w")
        b = tf.Variable(initial_value=tf.random_normal(shape=[10], mean=0.0, stddev=0.05), name="b")
        y_predict = tf.matmul(x, w) + b

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
            for i in range(2000):
                x_i, y_i = mnist_data.train.next_batch(50)
                sess.run(train_op, feed_dict={x: x_i, y_true: y_i})

                loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict={x: x_i, y_true: y_i})

                if (i + 1) % 20 == 0:
                    print('i=%d,loss=%f,accuracy=%f' % (i + 1, loss_i, accuracy_i))
                    saver.save(sess, './demo10_save')

        else:
            if os.path.exists('./demo10_save/checkpoint'):
                saver.restore(sess, './demo10_save')

            for i in range(100):
                x_i, y_i = mnist_data.test.next_batch(1)
                result_true = sess.run(y_predict, feed_dict={x: x_i, y_true: y_i})
                result_predict = sess.run(y_predict, feed_dict={x: x_i, y_true: y_i})

                true_num = tf.argmax(result_true, 1).eval()
                predict_num = tf.argmax(result_predict, 1).eval()

                print('第%d个样本，真实值=%d, 预测值=%d' % (i + 1, true_num, predict_num))


if __name__ == '__main__':
    mnist_demo()