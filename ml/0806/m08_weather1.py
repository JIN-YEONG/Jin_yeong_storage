from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

lr = LinearRegression(normalize=True)   # normalize -> 정규화(요소값 - 최소값)/ (최대값 - 최소값) , MinMaxScaler
lr.fit(train_x, train_y)
pre_y = lr.predict(test_x)
score = lr.score(test_x, test_y)
print(score)   # 0.936

# 시각화
plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()
    
rr = RandomForestRegressor(n_estimators=311,
                               max_depth=300,
                               min_samples_split=5,
                               min_samples_leaf=3,
                               max_features='auto'
)
rr.fit(train_x, train_y)
y_pred = rr.predict(test_x)
score = rr.score(test_x, test_y)   # 모델 검증 => model.evaluate()
print(score)   # 0.932

