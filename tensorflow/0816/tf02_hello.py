


import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")

'''
tf.constant() -> 특정 값을 넣어 텐서를 생성한다.
'''

sess = tf.Session()
print(sess.run(hello))