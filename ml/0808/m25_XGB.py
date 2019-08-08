from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
import pandas as pd

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';',encoding='utf-8')

# 데이터를 레이블과 데이터로 분리
y= wine['quality']
x= wine.drop('quality', axis=1)

# y레이블 변경하기
# y레이블의 값에 따라 세가지의 값으로 변경
newlist = []
for v in list(y):
    if v <=4:
        newlist += [0]   # newlist = newlist.append(0)
    elif v <= 7:
        newlist += [1]   # newlist = newlist.append(1)
    else:
        newlist += [2]   # newlist = newlist.append(2)

y = newlist
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


parameters ={
    'n_estimator': [10,100,150,200,250,300],
    'max_depth': [3,4,5,6,7,8,9,10,30,50,100]
}
kfold_cv = KFold(n_splits=10, shuffle=True)

search = RandomizedSearchCV(XGBClassifier(),
                            parameters,
                            n_iter=10,
                            n_jobs=-1,
                            cv = kfold_cv
)
search.fit(x_train, y_train)

print('최적의 매개변수 = ',search.best_params_)
print('최종 정답률=', search.score(x_test, y_test))

'''
최적의 매개변수 =  {'n_estimator': 150, 'max_depth': 8}
최종 정답률= 0.9479591836734694
'''