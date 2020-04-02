import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

'''
定义固定的超参数,方便待使用时直接传入。如果你问，这个超参数为啥要这样设定，如何选择最优的超参数？
这个问题此处先不讨论，超参数的选择在机器学习建模中最常用的方法就是“交叉验证法”。
另外，还要设置两个路径，第一个是数据下载下来存放的地方，一个是summary输出保存的地方。
'''
MODEL_SAVE_PATH = "model"  # 模型保存路径
MODEL_NAME = "mnist_model"  # 模型保存文件名
logdir = './graphs/mnist'  # 输出日志保存的路径
dropout = 0.6
learning_rate = 0.001
STEP = 100
# 每个批次的大小
batch_size = 200
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# create model
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)

    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('layer'):
    W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
    b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
    L1 = tf.nn.tanh(tf.matmul(x, W1) + b1, name='L1')
    L1_drop = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
    b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
    L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2, name='L2')
    L2_drop = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='W3')
    b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
    prediction = tf.matmul(L2_drop, W3) + b3

preValue = tf.argmax(prediction, 1,output_type='int32',name='output')

# 计算所有样本交叉熵损失的均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='loss')

optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step, name='train')

# 计算准确率
# 分别将预测和真实的标签中取出最大值的索引，若相同则返回1(true),不同则返回0(false)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求均值即为准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()  # 实例化saver对象
with tf.Session() as sess:
# 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)

    # 断点续训，如果ckpt存在，将ckpt加载到会话中，以防止突然关机所造成的训练白跑
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(STEP):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        opt, step = sess.run([optimizer, global_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        acc_train, step = sess.run([accuracy, global_step],
                                   feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})

        # 记录训练集的summary
        acc_test = sess.run([accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

        if i % 2 == 0:
            print("Iter" + str(step) + ", Testing accuracy:" + str(acc_test) + ", Training accuracy:" + str(acc_train))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)  # 保存模型
    #形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile("model/mnist.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())