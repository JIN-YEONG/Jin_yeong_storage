from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year= (df['연'] <= 2013)
test_year = (df['연'] >= 2014)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []   # 학습데이터
    y = []   # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval: continue

        y.append(temps[i])
        xa = []

        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)

    return(x,y)   # y=현재의 기온, x= y보다 과거 6일동안의 기온

x_train, y_train = make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

parameters ={
    'max_depth' : np.arange(1,10),#[3,4,5,6,7,8,9,10,20,30],
    'min_samples_leaf' : np.arange(1,10),#[1,2,3,4,5,6,7],
    'min_samples_split' : np.arange(2,10)#[2,3,4]
}

# print(parameters)

kfold_cv = KFold(n_splits=5, shuffle=True)

search = RandomizedSearchCV(DecisionTreeRegressor(),
                            parameters,
                            n_iter=10,
                            n_jobs=-1,
                            cv = kfold_cv
)
search.fit(x_train, y_train)

y_predict = search.predict(x_test)
print('최적의 매개변수 = ',search.best_params_)
print('훈련 정확도 =', search.score(x_train,y_train))
# print(search.best_estimator_)

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)
# print('score', search.score(x_test,y_test))

'''
최적의 매개변수 =  {'min_samples_split': 4, 'min_samples_leaf': 5, 'max_depth': 6}
훈련 정확도 = 0.9462991579212812
R2: 0.9264738165635574
'''