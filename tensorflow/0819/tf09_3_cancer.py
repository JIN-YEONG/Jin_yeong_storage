
# 유방암 numpy 파일을 이용하여 코딩

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.load('./data/npy_data/cancer_data.npy')   # 유방암 데이터셋
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]
# print(x_data.shape)   # 569,30
# print(y_data.shape)   # 569,1


x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([30,1]), name= 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# tf.random_normal(shape) -> shape에 맞는 랜덤한 수를 반환

logits = tf.matmul(x,w) + b
hypothesis = tf.sigmoid(logits)

# cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis))   # bineary_crossentropy
cost= tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy= tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y: y_data})

    j=0
    for i in h:
        if i == 1:
            j += 1     

    print('\n Hypothesis:', h , '\n Coreect (y):', c, '\n Accuracy:', a)

'''
Accuracy: 0.87873465
'''