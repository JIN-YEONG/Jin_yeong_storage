# 도마뱀 337페이지,342페이지

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV , KFold, RandomizedSearchCV
import numpy as np


iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

parameters = {
    'C': [1,10,100,1000], 
    'kernel': ['linear','rbf','sigmoid'], 
    'gamma': ['auto',0.001,0.0001]
}


# 랜덤 서치
kfold_cv = KFold(n_splits=5, shuffle=True)   # 데이터를 테스트와 검증 데이터로 분할
clf = RandomizedSearchCV(
    estimator=SVC(), 
    param_distributions=parameters,    # dict 타입의 데이터만 들어갈 수 있다.
    n_iter=10,   # epoch 값과 동일 -> 조합가능한 파라미터 중 10개의 값을 가져와 사용한다.
    cv=kfold_cv
)   

'''
GridSearch와 RandomSearch의 차이점 
GridSearch는 파라미터 값으로 생성 가능한 모든 모델을 실행하여 최적의 값을 얻음
RandomizedSearch는 파라미터 값의 모든 조합 중 정해진 개수를 선택하여 모델을 실행하여 최적의 값을 얻는다.

'''

clf.fit(x_train, y_train)   # KFold를 사용하기 때문에 나눠진 데이터를 사용할 필요가 없다고 생각된다.
print("최적의 매개변수= ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)