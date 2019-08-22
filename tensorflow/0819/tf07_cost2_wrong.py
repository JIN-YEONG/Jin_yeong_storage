

import tensorflow as tf

# tf Graph Input
x = [1,2,3]
y = [1,2,3]

# Set wrong model weights
w = tf.Variable(5.0)   # 잘못된 weight값
'''
w에 잘못된 초기 값을 넣더라도 훈련을 거듭하면서 올바른 w로 변한다.
'''

# Linear model
hypothesis = x * w

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Minimize: Gradient Descent Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, w_val = sess.run([train, w])
        print(step, w_val)

'''
0 5.0   <- 처음에는 w의 초기값이 들어간다.
1 1.2666664   <- 훈련된  w값이 들어간다.
2 1.0177778
3 1.0011852
4 1.000079
5 1.0000052
...
95 1.0
96 1.0
97 1.0
98 1.0
99 1.0
100 1.0
'''