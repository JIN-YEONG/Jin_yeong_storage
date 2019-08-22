


# Linear Regression
import tensorflow as tf

tf.set_random_seed(777)   # for reproducibility

w = tf.Variable(tf.random_normal([1]), name= 'weigth')   # tf.random_normal([1]) (1,1) shape의 렌덤한 수로 초기화
b = tf.Variable(tf.random_normal([1]), name = 'bias')

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w + b   # y = wx + b    모델 구성 

######### model.compile() #########
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis -y))   # loss= 'mse'
# tf.square(a) -> a^2
# tf.reduce_mean(a) -> a의 평균
# mse(Mean Square Error) -> (y_predict - y_data)^2의 값들의 평균

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)   # optimizer
###########################

# Launch the graph is a session
with tf.Session() as sess:   # Session은 사용후 close를 해야하는데 생략하기 위해 with 사용
    # Initializes global variable in the graph
    sess.run(tf.global_variables_initializer())   # 모든 변수의 초기화

    ######### model.fit() ############
    # Fit the line
    for step in range(2001):   #  epoch=2001
        _, cost_val, w_val, b_val = sess.run(   # cost_val -> 최소값이 최고   sess.run()에 train만 넣어도 된다.(교육용 코드이기 때문에 cost,w,b를 넣었다.)
            [train, cost, w, b], feed_dict={x: [1,2,3], y: [1,2,3]}   # feed_dict -> placeholder에 값을 넣는다.
        )
        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)
    ##################################
    
    ##### model.predict() #####
    # Testing our model
    print(sess.run(hypothesis, feed_dict={x: [5]}))   # 모델과 새로운 값을 넣는다.
    print(sess.run(hypothesis, feed_dict={x: [2.5]}))
    print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))
    '''
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    '''
    #########################



'''
Session().run()에 무엇을 넣느냐에 따라
fit가 될수도 있고 predict가 될 수 도 있다.

train을 넣을 경우 fit의 역할을 수행 cost를 같이 넣으면 최소 cost값을 확인 할 수 있다.
hypothesis 값을 넣으면 predict의 역할을 수행하는데 hypothesis값이 y값이기 때문이다.

'''

