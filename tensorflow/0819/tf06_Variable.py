
# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오

import tensorflow as tf
# tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [1,2,3]


w = tf.Variable(tf.random_normal([1]), name = 'weight')
b= tf.Variable(tf.random_normal([1]), name='bias')
print(w)   # w의 형태 출력  <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
print(b)   # b의 형태 출력  <tf.Variable 'bias:0' shape=(1,) dtype=float32_ref>

w = tf.Variable([0.3], tf.float32)   # w = 0.3

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa= sess.run(w)   
# print(aaa)   #   w의 값 출력  [0.3]
# sess.close()

# sess = tf.InteractiveSession()   
# sess.run(tf.global_variables_initializer())
# aaa= b.eval()   # Session.run(b)   [0.3]
# print(aaa)   # b의 값 출력   [-2.1405475]
# sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = w.eval(session=sess)   # session에서 eval사용하는 방법
print(aaa)   # [0.3]
sess.close()

'''
3가지의 실행 방법
1. Session(), run(w)
2. InteractiveSession(), w.eval()
3. Session(), w.eval(session=Session())

3가지 방법 모두 동일한 실행 방법이다.
'''