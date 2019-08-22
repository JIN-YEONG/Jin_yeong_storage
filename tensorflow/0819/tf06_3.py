


import tensorflow as tf

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# model parameters
w = tf.Variable([0.3], tf.float32)   # 처음 선언 할때는 아무 값이나 들어가 있어도 된다.
b = tf.Variable([-0.3], tf.float32)  # 훈련의 과정 속에서 훈련된 가중치로 변경되기 때문이다.

# model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * w + b

# cost/loss function
cost = tf.reduce_sum(tf.square(hypothesis - y))   # SSE(Sum of Square Error)

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
# train = optimizer.minimize(cost)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):   # epoch= 1000
        sess.run(train, {x: x_train, y: y_train})

    w_val ,b_val, cost_val = sess.run([w,b,cost], feed_dict={x:x_train, y: y_train})   # 마지막 훈련에 대한 

    print(f'W: {w_val} b: {b_val} cost:{cost_val}')   
    # 포멧 문자열 리터럴
    # 문자열 앞에 f를 붙이는 표현식
    # {} 안에 변수명을 넣어 바로 출력 할 수 있다.

'''
W: [-0.9999969] b: [0.9999908] cost:5.699973826267524e-11
'''