# 도마뱀 337페이지,342페이지

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV , KFold



iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

parameters = [   # 튜닝할 파라미드터의 값
    {'C': [1,10,100,1000], 'kernel': ['linear']},
    {'C': [1,10,100,1000], 'kernel': ['rbf'], 'gamma':[0.001,0.0001]},
    {'C': [1,10,100,1000], 'kernel': ['sigmoid'], 'gamma':[0.001, 0.0001]}
]


# 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)   # 데이터를 테스트와 검증 데이터로 분할
clf = GridSearchCV(
    estimator=SVC(),   # 사용할 모델
    param_grid=parameters,   # dict, list, 여러개의 dict가 들어올 수 있다.
    cv=kfold_cv   # 교차검증 방법
)

'''
GridSearchCV()
파라미터 값들로 만들수 있는 모든 모델을 실행하여 최적의 파라미터 값을 찾는다.

'''

clf.fit(x_train, y_train)   # KFold를 사용하기 때문에 나눠진 데이터를 사용할 필요가 없다고 생각된다.
print("최적의 매개변수= ", clf.best_estimator_)   # 최적의 파리미터값을 출력

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)