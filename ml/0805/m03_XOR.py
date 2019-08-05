from sklearn.svm import LinearSVC, SVC   # Linear 선형
from sklearn.metrics import accuracy_score   # 정확도

# 1. 데이터
# AND 값
x_data = [[0,0], [1,0], [0,1], [1,1]]   
y_data = [0,1,1,0]                    

# 2. 모델
# model = LinearSVC()   # 머신러닝 회귀모델
model = SVC()

'''
SVC
값을 3차원으로 만들어 W값을 구한다.

LinearSVC는 선형회귀방식을 사용하기 때문에 정확도가 낮다.
SVC는 데이터를 3차원으로 바꿔 계산하기 때문에 정확도가 높다.

XOR 서로 값이 다를떄 1
XOR 계산의 경우 평면에서는 만족스러운 W값을 구할수 없기 때문에
3차원에서 W값을 구하는 SVC모델를 사용한다.
'''

# 3. 실행
model.fit(x_data, y_data)         

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]  
y_predict = model.predict(x_test)


print(x_test, '의 예측 결과: ', y_predict)
print('acc= ', accuracy_score(y_data, y_predict))   # y_data와 y_predict가 얼마나 일치하는지 비교

