

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('./data/csv_data/data-01-test-score.csv', delimiter=',', dtype= np.float32)
x_data = xy[:, 0:-1]   # 마지막 열을 제외한 모든 열
y_data = xy[:,[-1]]   # 마지막 열
# print(x_data, '\nx_data shape:', x_data.shape)   # x_data.shape = (25,3)
# print(y_data, '\ny_data shape:', y_data.shape)   # y_data.shape = (25,1)

x = tf.placeholder(tf.float32, shape=[None, 3])   # input_dim =3 
y = tf.placeholder(tf.float32, shape=[None, 1])   # output=1

w = tf.Variable(tf.random_normal([3,1]), name='weight')   # input=3, output=1
b = tf.Variable(tf.random_normal([1]), name='bias')   # output=1

hypothesis = tf.matmul(x,w) + b   # wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val ,hy_val, _ = sess.run(
        [cost, hypothesis, train],   
        feed_dict={x:x_data, y:y_data}   # shape: x->(None,3)   y->1
    )

    if step % 10 == 0:
        print(step, 'Cost:', cost_val, '\nPrediction\n', hy_val)   # hy_val은 예측값을 출력 -> y값과 비슷해야 한다.

'''
2000 Cost: 24.722485
Prediction
 [[154.42894 ]
 [185.5586  ]
 [182.90646 ]
 [198.08955 ]
 [142.52043 ]
 [103.551796]
 [146.79152 ]
 [106.70152 ]
 [172.15207 ]
 [157.13037 ]
 [142.5532  ]
 [140.17581 ]
 [190.05006 ]
 [159.59953 ]
 [147.35217 ]
 [187.26833 ]
 [153.3315  ]
 [175.3862  ]
 [181.3706  ]
 [162.1332  ]
 [172.44307 ]
 [173.06042 ]
 [164.7337  ]
 [158.24257 ]
 [192.79166 ]]
'''