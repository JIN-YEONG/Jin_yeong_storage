from sklearn.svm import LinearSVC, SVC   # Linear 선형
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score   # 정확도

# 1. 데이터
# AND 값
x_data = [[0,0], [1,0], [0,1], [1,1]]   
y_data = [0,1,1,0]                    

# 2. 모델 (XOR 분류)
# model = LinearSVC()   # 머신러닝 회귀모델
model = KNeighborsClassifier(n_neighbors=1)   # 근접데이터를 비교하여 분류 수행
                                              # n_neighbors = 비교할 인접 값의 수

'''
KNN 모델(K-Nearest Neighbors)
최근접 이웃 모델

KNeighborsClassifier
최근접 이웃 분류모델 
새로운 데이터가 들어왔을때 그 데이터가 어느 그룹에 속하는지 분류하기 위해 
인접한 값과 비교하여 분류하는 모델
k(n_neighbors)개의 인접한 데이터를 비교한다.

KNeighborsRegressor
최근접 이웃 회귀 모델
테스트 데이터를 x축에 세워 가장 인접한 값의 y값을 갖는다.
k(n_neighbors)값을 늘리면 인접한 값의 평균으로 y값을 갖는다.
'''

# 3. 실행
model.fit(x_data, y_data)         

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]  
y_predict = model.predict(x_test)


print(x_test, '의 예측 결과: ', y_predict)
print('acc= ', accuracy_score(y_data, y_predict))   # y_data와 y_predict가 얼마나 일치하는지 비교

