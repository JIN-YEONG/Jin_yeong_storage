

# Multi-variable linear regression2
import tensorflow as tf
tf.set_random_seed(777)

# 데이터
#          x1   x2   x3
x_data = [[73., 80., 75.],   # (5,3)
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.],   # 5,1
          [185.],
          [180.],
          [196.], 
          [142.]]


# 텐서 모델
# placehodlers for a tensor that will be always fed
x = tf.placeholder(tf.float32, shape=[None, 3])   # input_dim = 3   x의 calumn의 개수
y = tf.placeholder(tf.float32, shape=[None, 1])   # output = 1   y의 calumn의 개수
# shape=[None, 1]에서의 None -> 행 무시

w = tf.Variable(tf.random_normal([3, 1]), name='weight')   # tf.random_normal([3,1]) => input_dim=3, output = 1
b= tf.Variable(tf.random_normal([1]), name='bias')   # tf.random_normal([1]) => output=1

hypothesis = tf.matmul(x,w) + b   # tf.matmul() -> 행렬의 곱 (x1*w1 + x2*w2 + x3*w3)
'''
  x1   x2   x3      *      w      =   tf.matmul()
[[73., 80., 75.],        [[w1],      [[73*w1 + 80*w2 + 75*w3],
 [93., 88., 93.],                     [93*w1 + 88*w2 + 93*w3],
 [89., 91., 90.],   *     [w2],   =   [89*w1 + 91*w2 + 90*w3],
 [96., 98., 100.],                    [96*w1 + 98*w2 + 100*w3],
 [73., 66., 70.]]         [w3]]       [73*w1 + 66*w2 + 70*w3]]
'''

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
        feed_dict={x:x_data, y:y_data}   
    )

    if step % 10 == 0:
        print(step, 'Cost:', cost_val, '\nPrediction\n', hy_val)   # hy_val은 예측값(y_predict)을 출력 -> y값과 비슷해야 한다.

'''
2000 Cost: 3.178887
Prediction
 [[154.3593 ]
 [182.95117]
 [181.85052]
 [194.3554 ]
 [142.03566]]
'''