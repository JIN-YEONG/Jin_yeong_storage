# 도마뱀 76페이지

# 선형 회귀 모델

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()

# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# print(type(boston))

from sklearn.linear_model import LinearRegression, Ridge, Lasso

model1 = LinearRegression()  # 선형 회귀 모델
model1.fit(x_train, y_train)
result = model1.score(x_test, y_test)
print(result) # 0.78

model2 =Ridge(alpha=1.0)   # L2규제를 사용하는 선형회귀모델
                           # alpha -> 훈련세트의 성능대비 모델을 얼마나 단순화 할지 결정
'''
l2 규제 
가중치(w)의 절대값을 가능한 작게 만드는 규제
직선의 기울기가 작아진다.
'''
model2.fit(x_train, y_train)
result= model2.score(x_test, y_test)
print(result)   # 0.77


model3 =Lasso(alpha=1.0)   # L1규제를 사용하는 선형회귀 모델
                           # alpha -> 훈련세트의 성능대비 모델을 얼마나 단순화 할지 결정
'''
l1 규제
가중치(w)를 0에 가깝게 만드는 규제
l2규제와 달리 실재로 0이되는 w도 존재한다.
모델이 이해하기 쉬워지고 가장 중요한 특징이 무엇인지 드러내 준다.
'''
model3.fit(x_train, y_train)
result = model3.score(x_test, y_test)
print(result)   # 0.68

# y_pred = model.predict(x_test)
# print(y_pred)