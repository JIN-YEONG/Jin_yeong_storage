
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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

# 학습하기
model = RandomForestClassifier(n_estimators=300,
                               max_depth=100,
                               min_samples_split=11,   # 내부 노드를 분배하는 필요한 최소한의 샘플 수
                               min_samples_leaf=1,   # 리프노드(가장 아래쪽 노드)에 있어야할 최소한의 샘플 수 
                               max_features='auto'
)

''' 

'''

model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)   # 정확도와 같은 값이 나온다. R2값

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))   # 결과 리포트 출력
print('정답률=', accuracy_score(y_test, y_pred))
print(aaa)


'''
keras의 대략적 흐름
model = Sequential() / Models

model.fit(x_train, y_train, metrics=['acc'/'mse'/'mae'])

loss, acc/mse/mae = model.evaluate(x_test, y_test)

y_pred - model.predict(new_x)


ML의 대략적 흐름
model = RandomForestClassifier()
model.fit(x_train, y_train)
# 데이터 평가
result = model.score(x_test, x_test)   # 검증 keras의 model.evaluate()와 같음

y_pred = model.predict(new_x)
'''