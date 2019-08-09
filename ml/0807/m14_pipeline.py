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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])   # 전처리와 모델의 이름을 정해주어야 한다.
                # 전처리                       모델
# from sklearn.pipeline import make_pipeline
# pipe = make_pipeline(MinMaxScaler(), SVC())   # 별도의 이름 지정 없이 사용가능
                                                # 파라미터에서는 이름을 지정해 주어야 한다.(클래스의 소문자 이름)


pipe.fit(x_train,y_train)

print('테스트 점수: ', pipe.score(x_test, y_test))


'''
도마뱀 396페이지 근처
k fold만 사용했을 때의 문제점
train데이터에서 val데이터가 만들어 지기 때문에 검증결과가 과적합된다.

pipeline을 사용하면 kfold로 나눠진 데이터에서 train데이터에만 전처리 작업을 수행할 수 있다.
train데이터와 val데이터가 다르기때문에 과적합을 해결 할 수 있다.


pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
pipe.fit(x_train,y_train)
pipline이 fit되면 들어온 데이터가 MinMaxScaler로 전처리 되고 SVC모델로 들어가 모델이 실행 된다.
'''