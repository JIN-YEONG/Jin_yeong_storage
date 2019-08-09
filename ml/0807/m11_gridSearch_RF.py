import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV , KFold



iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

parameters = [   # RandomForestClassifier의 파라미터
    {'n_estimators': [10,100,300,500],
     'max_depth': [10,100,200],
     'min_samples_split' : [0.1,0.2,0.5,0.8],
     'min_samples_leaf' : [0.1,0.2,0.3]
    }
]

# RandomForestClassifer 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)


clf.fit(x_train, y_train)   # KFold를 사용하기 때문에 나눠진 데이터를 사용할 필요가 없다고 생각된다.
print("최적의 매개변수= ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)