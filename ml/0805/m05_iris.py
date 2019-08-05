
'''
LinearRegression, Ridge, Lasso
-> 선형 모델 (y=wx+b) 선을 만드는 모델
=> LinearSVC, KNeighborsRegressor
Ridge -> l2
Lasso -> l1

'''

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC , LinearSVC
from sklearn.metrics import accuracy_score


iris_data = pd.read_csv('./data/iris.csv', 
                        names=['a','b','c','d','y'], 
                        encoding='utf-8')
# print(iris_data)
# print(iris_data.shape)
# print(type(iris_data))

# 열의 이름을 이용한 자르기
y = iris_data.loc[:, "y"]
x= iris_data.loc[:, ['a', 'b', 'c', 'd']]
# # 열의 순서를 이용한 자르기
# y2 = iris_data.iloc[:,4]
# x2 = iris_data.iloc[:,0:4]

# print(x.shape)   # (150,4)
# print(y.shape)   # (150,)


# 학습전용과 테스트 전용 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.7, shuffle=True)
# test_size와 train_size의 합이 1보다 작아도 자를 수 있다. (남는 데이터는 버림)(1보다 크면 안됨)


# 학습하기
# clf = SVC()   # 0.9666666
# clf = KNeighborsClassifier(n_neighbors=1)   # 0.9
clf = LinearSVC()   # 1.0

# 실행
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답율: ", accuracy_score(y_test, y_pred))
