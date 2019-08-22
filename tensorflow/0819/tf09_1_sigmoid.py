

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],   # (6,2)
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],   # (6,1)
          [0],
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])   # input_dim=1
y = tf.placeholder(tf.float32, shape=[None, 1])   # output=1

w = tf.Variable(tf.random_normal([2,1]), name='weight')   # input:2  output:1
b = tf.Variable(tf.random_normal([1]), name='bias')   # output=1

# Hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)   # 기존의 hypothesis를 sigmoid형식으로 바꾸기 위해 tf.sigmoid()에 넣어준다.

# 로지스틱 리스레션에서 cost에 -가 붙는다. (로그 값이 -무한대를 막기위해)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))   # bineary_crossentropy 계산식(일단 통체로 암기)
# cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(x,w)+b, labels=y_data)   

'''
수식을 이용한 bineary_crossentropy의 경우 tf.log(x)의 x가 0이 되면 음의 무한대가 되기 때문에 값이 없어진다.
x를 tf.clip_by_value(x, 1e-4,1)을 이용하여 범위를 지정해 주는 것으로 예방이 가능하다.

반면 함수를 사용하면 이러한 번거로움 없이 결과를 출력할 수 있다.
'''

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 이진 분류에서만 사용가능한 predicted와 accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)   # hypothesis가 0.5 이상이면 cast
# tf.cast() -> 1. 실수를 버림하여 정수로 바꾼다.
#           -> 2. 조건이 True면 1, False면 0

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))   # 이진 분류에서만 사용가능
# tf.euqal(x,y)  x와 y를 비교하여 boolean값 반환

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):   # 100회 했을 때 acc:0.5 , 1000회 -> acc:0.833
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})   # h -> y값, c-> predict, a->accuracy
    print('\nHypothesis:', h, "\nCorrenct (y):", c, "\nAccuracy:", a)

'''
Hypothesis: [[0.2305116 ]
 [0.2698999 ]
 [0.69877255]
 [0.6247256 ]
 [0.7793784 ]
 [0.92801166]]
Correnct (y): [[0.]
 [0.]
 [1.]
 [1.]
 [1.]
 [1.]]
Accuracy: 0.8333333
'''