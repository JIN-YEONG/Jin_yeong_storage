# m11_randomSearch.py에 pipeline을 적용

# 도마뱀 337페이지,342페이지

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV , KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np


iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

parameters = {
    'svc__C': [1,10,100,1000], 
    'svc__kernel': ['linear','rbf','sigmoid'], 
    'svc__gamma': ['auto',0.001,0.0001]
}

'''
pipline 안의 모델에 접근하기 위해 
'모델명__파라미터명'의 형태의 key값으로 파라미터 값들을 만들어야 한다.

Pipeline()을 사용했을때는 사용자가 지정한 모델명을 
make_pipeline()을 사용했을경우 소문자로된 모델클래스명을 모델명으로 사용한다.
'''

# 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)   # 데이터를 테스트와 검증 데이터로 분할
# clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)

from sklearn.pipeline import make_pipeline
pipe= make_pipeline(MinMaxScaler(), SVC())
# pipe = Pipeline( [ ('scaler', MinMaxScaler()), ('svm', SVC()) ] )
# pipe.fit(x_train,y_train)

clf = RandomizedSearchCV(
    estimator=pipe, 
    param_distributions=parameters,    # dict 타입의 데이터만 들어갈 수 있다.
    n_iter=10, 
    cv=5
)   

clf.fit(x_train, y_train)  
'''
clf.fit()으로 인해 RandomizedSearchCV가 실행되고 
cv값에 의해 x_train이 cross_validation 된다
그후 pipeline에 의해 train_data만 정규화가 진행되고 
정규화된 train_data가 모델에 들어가 실행된다.
'''
print("최적의 매개변수= ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)