

# Multi-variable linear regression 1
import tensorflow as tf
tf.set_random_seed(777)   # for reproducibility

# 다중 입력
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142]

# x의 개수만큼 placeholder를 생성
x1= tf.placeholder(tf.float32)
x2= tf.placeholder(tf.float32)
x3= tf.placeholder(tf.float32)

y= tf.placeholder(tf.float32)

# x가 3개라 w도 3개
# 각 x에 대한 가중치 w
# tf.random_normal([1])에서 1은 input_dim=1을 의미
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = (x1*w1 + x2*w2 + x3*w3) + b   # y = xw + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val ,hy_val, _ = sess.run(   # cost_val -> loss,  hy_val -> y_predict
        [cost, hypothesis, train],   
        feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data}
    )

    if step % 10 == 0:
        print(step, 'Cost:', cost_val, '\nPrediction\n', hy_val)   # hy_val은 예측값을 출력 -> y값과 비슷해야 한다.

sess.close()   # with를 사용하면 생략가능

'''
1000 Cost: 8.234499
Prediction
 [147.18959 187.54395 179.33307 195.61714 145.31201]
'''