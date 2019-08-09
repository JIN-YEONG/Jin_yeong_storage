from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

train_year = (df['연'] <= 2014)
test_year = (df['연'] >= 2015)

train_data = df.loc[df['연'] <= 2014, :]
test_data = df.loc[df['연'] >= 2015, :]
# print(train_data)


interval =6
def make_data(data):
    x=[]
    y=[]
    temp = list(data['기온'])

    for i in range(len(temp)):
        if i < interval:
            continue
        y.append(temp[i])

        xa=[]
        for p in range(interval):
            d = i - interval + p 
            xa.append(temp[d])
        x.append(xa)

    return(x,y)

x_train, y_train = make_data(train_data)
x_test, y_test = make_data(test_data)

# print(x_train[:10])
# print(y_train[:10])

model1 = RandomForestRegressor(n_estimators=311,
                               max_depth = 300,
                               min_samples_split=5,
                               min_samples_leaf =3,
                               max_features='auto'
)

model1.fit(x_train,y_train)
print('RF:', model1.score(x_test,y_test))

model2 = LinearRegression()
model2.fit(x_train,y_train)
print('LR:', model2.score(x_test,y_test))

model3 = Ridge(alpha=1.0)
model3.fit(x_train,y_train)
print('Ridge:', model3.score(x_test,y_test))

model4 = Lasso(alpha=1.0)
model4.fit(x_train, y_train)
print('Lasso:', model4.score(x_test,y_test))

'''
RF: 0.9253892707512781
LR: 0.9317681246004814
Ridge: 0.9317687474117703
Lasso: 0.9312990601192239
'''