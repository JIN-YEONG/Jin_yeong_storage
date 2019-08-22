


# Linear Regression
import tensorflow as tf

tf.set_random_seed(777)   # for reproducibility

w = tf.Variable(tf.random_normal([1]), name= 'weigth')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w + b   # y = wx + b    모델 구성 

######### model.compile() #########
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis -y))   # loss='mse'


# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)   # optimizer
###########################

# Launch the graph is a session
with tf.Session() as sess:
    # Initializes global variable in the graph
    sess.run(tf.global_variables_initializer())   # w,b의 초기화

    ######### model.fit() ############
    # Fit the line
    for step in range(2001):   #  epoch=2001
        _, cost_val, w_val, b_val = sess.run(   # cost_val -> 최소값이 최고   sess.run()에 train만 넣어도 된다.(교육용 코드이기 때문에 cost,w,b를 넣었다.)
            [train, cost, w, b],
            feed_dict={x: [1,2,3,4,5], y: [2.1, 3.1, 4.1, 5.1, 6.1]}   # feed_dict -> placeholder에 값을 넣는다.
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
    [6.1009192]
    [3.5992656]
    [2.5986042 4.599927 ]
    '''
    #########################

    '''
    sess.run()은 그래프를 텐서머신에 넣어 결과를 출력하는 것이다.
    넣는 값에 따라 fit, evaluate의 역활을 수행한다. 
    train값을 넣으면 fit
    hypothesis값을 넣으면 evaluate/predict
    '''