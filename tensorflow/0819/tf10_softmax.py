

# softmax classifier
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2,1,1,],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.placeholder('float', [None,4])
y = tf.placeholder('float', [None,3])
nb_classes = 3

w = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)   # softmax()

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))   # categorical_crossentropy 의 계산식

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)   # train

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    print('---------------------')
    # 새로운 값 출력
    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))

    print('---------------------')
    b = sess.run(hypothesis, feed_dict={x: [[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))

    print('---------------------')
    c = sess.run(hypothesis, feed_dict={x:[[1,1,0,1]]})
    print(c, sess.run(tf.argmax(c,1)))

    print('---------------------')
    all = sess.run(hypothesis, feed_dict={x: [[1,11,7,9], [1,3,4,3,], [1,1,0,1]]})
    print(all, sess.run(tf.argmax(all,1)))

'''
tf.argmax(a,1) -> a의 각 행에서 가장 큰 값을 갖는 열의 인덱스 반환
               -> one-hot_encoding을 풀때 사용한다.
tf.argmax(a,0) -> a의 각 열에서 가장 큰 값을 갖는 행의 인덱스를 반환


[[1.3890490e-03 9.9860185e-01 9.0612921e-06]] [1]
---------------------
[[0.9311919  0.06290223 0.00590592]] [0]
---------------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
---------------------
[[1.3890478e-03 9.9860197e-01 9.0612930e-06]
 [9.3119192e-01 6.2902197e-02 5.9059118e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]

'''
