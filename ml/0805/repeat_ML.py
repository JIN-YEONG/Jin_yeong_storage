from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

'''ML을 이용한 이진 분류 예제'''

dataset = pd.read_csv('./data/pima-indians-diabetes.csv',
                      encoding='utf-8', header=None)   # 첫번째 행을 열로 읽지 않고 csv읽어오기
# print(dataset.shape)   # 768,9

# 열의 순서를 이용한 분리
x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, test_size=0.3)   # 데이터 분할
# print(x_train.shape)   # 537,8
# print(y_train.shape)   # 537

# 여러가지 머신러닝 모델 (가장 높은 정확도 사용)
# model = SVC()   # 0.6493
# model = LinearSVC()   # 0.6623
model = KNeighborsClassifier(n_neighbors=1)   # 0.7142

# model = KNeighborsRegressor(n_neighbors=1)   # 0.6233


model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
print('acc:', accuracy_score(y_test, y_pred))   
# y_test와 y_pred를 단순비교하여 얼마나 일치하는 지 반환
# 단순 비교이기 때문에 분류모델에서만 정확한 비교가 가능