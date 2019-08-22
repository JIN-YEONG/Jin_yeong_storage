

import tensorflow as tf

tf.set_random_seed(777)

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name='weight')   # 초기값이 있어야 하기 때문에 random하게 1개의 값을 넣었다.
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * w + b   # wx + b   훈련시켜서 나오는 y값
# 훈련된 모델에서 나오는 새로운 값(y_predict)


# keras의 compile부분
cost = tf.reduce_mean(tf.square(hypothesis - y_train))   # loss= 'mse'
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)   # optimizer
#         경사하강법(Gradient descent)을 이용해서 cost를 최저의 값으로 만들어라


with tf.Session() as sess:   # Session은 만들면 close()를 해야하는데 with를 사용하면 close()가 없어도 된다.

    sess.run(tf.global_variables_initializer())   # 모든 변수 초기화 -> 모델의 다른 연산을 실행하기전에 반드시 명시적으로 사용하는 부분

    
    for step in range(2001):   # epochs=2001
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])   # model.fit()
        # 넣어준 값만큼 변수를 만든다.

        if step % 20 == 0:   # 20번 마다 출력
            print(step, cost_val, w_val, b_val)   # cost는 mse값이기 때문에 모델이 잘 되었는지 확인할 수 있다.




'''
카라스가 텐서플로우를 함수화 하여 만든 것이기 때문에 
텐서플로우에는 당연히 케라스의 기능이 있다.
대신 직접 지정해 주어야하는 것이 더 많고 그로인해 더 복잡하다.
기본적으로 텐서라는 일종의 노드를 트리형식으로 연결하여 모델을 구성하고
세션을 이용해 데이터와 결합하여 훈련이 진행된다. 세션런에 넣는 값에 따라 
카라스의 fit, predict, evaluate의 기능을 수행 할 수 있다.
'''