


import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)   # constant 상수
node2 = tf.constant(4.0)   # 실수형이기 때문에 4.0으로 사용
node3 = tf.add(node1, node2)   

'''
1. 3.0값을 가진 node1생성
2. 4.0값을 가진 node2생성
3. node1과 node2를 연결한 add생성(트리형태)

node를 print하면 node의 형태(그래프)가 출력

node를 Session()에 넣고 run()하면 노드의 값이 나온다.

텐서플로우는 각 텐서들의 트리 모양의 그래프로 생성되어 연결된다.
텐서자체를 print 한다면 텐서의 형태가 출력되고
Session()에 넣고 run()을 실행 시키면 텐서가 가지고 있는 값이 출력된다.
'''

# 데이터의 형이 출력 (값이 출력되지 않는다.)
print('node1:', node1, 'node2:', node2)
print('node3:', node3)

'''
1         -> 스칼라(한개만 있는 숫자)
(1,2)     -> 백터(1행 n열)   input_dim=(1,)
[[1,2]]   -> 행렬   input_shape=(2,3)
........  -> 텐서(행렬이상)
'''

# node를 Session()에 넣고 run()하면 노드의 값이 나온다.(keras의 fit 역할)
sess = tf.Session()
print('sess.run(node1, node2):', sess.run([node1, node2]))
print('sess.run(node3):', sess.run(node3))

sess.close()

'''
y = wx+b
1. w노드 생성
2. x노드 생성
3. b노드 생성
4. w와x를 곱하는 mul노드 생성
5. mul노드와 b를 더하는 add노드 생성
6. 생성된 노드들을 Session에 넣고 run하면 y값이 나온다.
'''