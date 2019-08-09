import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV , KFold


iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

parameters = [   # KNeighborsClassifier의 파리미터
    {'n_neighbors': [1,3,5,7]}
]

# KNeighborsClassifier 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)   # 데이터를 테스트와 검증 데이터로 분할
clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=kfold_cv)


clf.fit(x_train, y_train)   
print("최적의 매개변수= ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)