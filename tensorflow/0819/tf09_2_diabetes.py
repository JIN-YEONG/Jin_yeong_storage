
# Logistic Regression Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy= np.loadtxt('./data/csv_data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)   # 인디언 당료병 데이터
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# print(x_data.shape, y_data.shape)   # (759, 8) (759, 1)

# placeholder for a tensor that will be always fed
x = tf.placeholder(tf.float32, shape=[None, 8])   # 8개의 열   input_dim=8
y = tf.placeholder(tf.float32 ,shape=[None, 1])   # 1개의 열   output=1

w = tf.Variable(tf.random_normal([8,1]), name= 'weight')   # input: 8   output: 1   # shape를 맟춰 주어야 한다.
b = tf.Variable(tf.random_normal([1]), name= 'bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(x,w)))
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# cost/loss function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1- hypothesis))   # bineary_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))   # 이진 분류에서만 사용가능

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y:y_data})

        if step % 200 == 0:
            print(step,cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})

    print('\nHypothesis:', h, '\n Correct (y):', c, '\nAccuracy:',a)

'''
Accuracy: 0.7628459
'''