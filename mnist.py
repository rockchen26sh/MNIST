# coding = utf-8

# batch_size = 200                                  #单批次数据量
# hidden_layer1_node = 500                         #隐藏层1节点数
# hidden_layer2_node = 300                         #隐藏层2节点数
# output_layer_node = 10                            #输出层节点数
# steps = 500                                        #训练迭代次数
# lr = 0.1                                          #学习速率
# regularization_rate = 0.0015                       #正则化系数

def nn_train(batch_size, hidden_layer1_node, hidden_layer2_node, output_layer_node, steps, lr, regularization_rate):

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    # Load dataset
    mnist = input_data.read_data_sets('./dataset/MNIST_data', one_hot=True)

    # 插入点
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, output_layer_node])
        keep_prob = tf.placeholder(tf.float32)

    # 变量
    with tf.name_scope('Var'):
        w1 = tf.Variable(tf.truncated_normal(shape=[784, hidden_layer1_node], stddev=0.1))
        w2 = tf.Variable(tf.truncated_normal(shape=[hidden_layer1_node, hidden_layer2_node], stddev=0.1))
        w3 = tf.Variable(tf.truncated_normal(shape=[hidden_layer2_node, output_layer_node], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer1_node]), name='b1')
        b2 = tf.Variable(tf.constant(0.1, shape=[hidden_layer2_node]), name='b2')
        b3 = tf.Variable(tf.constant(0.1, shape=[output_layer_node]), name='b3')

    with tf.name_scope('layer'):
        # 神经网络结构
        with tf.name_scope('hidden_layer1'):
            hidden_layer1 = tf.matmul(x, w1) + b1
            hidden_layer1 = tf.nn.relu(hidden_layer1)
        hidden_layer1 = tf.nn.dropout(x=hidden_layer1, keep_prob=keep_prob)
        with tf.name_scope('hidden_layer2'):
            hidden_layer2 = tf.matmul(hidden_layer1, w2) + b2
            hidden_layer2 = tf.nn.relu(hidden_layer2)
        hidden_layer2 = tf.nn.dropout(x=hidden_layer2, keep_prob=keep_prob)
        with tf.name_scope('out_layer'):
            out_layer = tf.matmul(hidden_layer2, w3) + b3

    # 正则化
    regular = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regular(w1) + regular(w2) + regular(w3)

    # Cost Function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_layer)) + regularization
        tf.summary.scalar('loss',loss)

    # optimi
    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # accuracy cal
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(out_layer),1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./tensorborad/logs/train',tf.get_default_graph())
        train_writer = tf.summary.FileWriter('./tensorborad/logs/train')
        for i in range(steps):
            x_train, y_train = mnist.train.next_batch(i)
            sess.run(train_step, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
            if (i+1) % 5 == 0:
                summary,acc = sess.run([merged,accuracy], feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
                train_writer.add_summary(summary,i)
                print('Iter %d, Testing Accuracy: %f' % (i + 1, acc))
            cost = sess.run(loss, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    return cost,acc

# batch_size = 200                                  #单批次数据量
# hidden_layer1_node = 500                         #隐藏层1节点数
# hidden_layer2_node = 300                         #隐藏层2节点数
# output_layer_node = 10                            #输出层节点数
# steps = 500                                        #训练迭代次数
# lr = 0.1                                          #学习速率
# regularization_rate = 0.0015                       #正则化系数

#print(nn_train(2000,hidden_layer1_node,hidden_layer2_node,output_layer_node,steps,lr,regularization_rate))
from hyperopt import hp, tpe, fmin,partial
space = {hidden_layer1_node:hp.choice("hidden_layer1_node",range(1,10)) * 10,
         hidden_layer2_node:hp.choice("hidden_layer2_node",range(1,10)) * 10,
         lr:hp.uniform("lr",0.0001,0.001),
         regularization_rate:hp.uniform('regularization_rate',0.00001,0.0001)}

def NN(argsDict):
    batch_size = 200
    hidden_layer1_node = argsDict['hidden_layer1_node']
    print ('hidden_layer1_node:' + hidden_layer1_node)
    hidden_layer2_node = argsDict['hidden_layer2_node']
    output_layer_node = 10
    lr = argsDict['lr']
    regularization_rate = argsDict['regularization_rate']
    steps = 500

    nn = nn_train(batch_size, hidden_layer1_node, hidden_layer2_node, output_layer_node, steps, lr, regularization_rate)
    return -nn

algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(NN,space,algo=algo,max_evals=4)



