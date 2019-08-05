from sklearn.svm import LinearSVC, SVC   # Linear 선형
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score   # 정확도
import numpy as np

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")   # 경로의 './' 은 현재 경로(study)
                                                                          # 경로의 '../'은 상위 폴더

X = dataset[:,0:8]
Y = dataset[:,8]

 
# 모델 설정 (이진 분류모델)
''' 가장 결과가 좋은 모델을 사용하면 된다.'''
# model = LinearSVC()
model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)

#############################################
# model = KNeighborsRegressor(n_neighbors=1)   
# n_neighbors=1이기 때문에 이진 분류 모델에서 사용가능(n_neighbors값이 달라지면 사용 불가)
# 회귀 모델 이기떄문에 바람직한 사용법은 아니다.


# 모델 실행
model.fit(X,Y)

y_predict = model.predict(X)
print(y_predict)
# 결과 출력
print('\n Accuracy : %.4f' % (accuracy_score(Y,y_predict)))