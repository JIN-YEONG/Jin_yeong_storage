


import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)   # constant 상수
node2 = tf.constant(4.0)   # 실수형이기 때문에 4.0으로 사용
node3 = tf.add(node1, node2)   

# 데이터의 형이 출력 (값이 출력되지 않는다.)
# print('node1:', node1, 'node2:', node2)
# print('node3:', node3)

# node를 Session()에 넣고 run()하면 노드의 값이 나온다.(keras의 fit 역할)
sess = tf.Session()
# print('sess.run(node1, node2):', sess.run([node1, node2]))
# print('sess.run(node3):', sess.run(node3))

a = tf.placeholder(tf.float32)   # placeholder -> 나중에 값을 넣음(sess에서 feed_dict를 이용)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))   # feed_dict를 이용해 placeholder에 값을 넣음
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))   # list를 넣을 수 있다.

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))

'''
keras와 달리 tensorflow는 그래프의 모양을 생각해야한다.

탠서 생성 방법
1. tf.constant(x) -> x이 들어 있는 상수를 선언
2. tf.placeholder(dtype) -> 지정된 데이터 타입의 공간을 생성한다. run()의 feed_dict속성을 이용해 값을 넣을 수 있다.
3. tf.Variable(x) -> x값으로 초기화된 변수를 선언

'''

