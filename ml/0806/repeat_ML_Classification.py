from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')

# count_data = wine.groupby('quality')['quality'].count()
# print(count_data)   # y값의 종류별 개수

y = wine['quality']
x = wine.drop('quality', axis=1)
# print(y.values)

newlist=[]
for v in y.values:
    if v <= 4:
        newlist.append(0)
    elif v<=7:
        newlist.append(1)
    else:
        newlist.append(2)

y = newlist

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100,
                               max_depth =100,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               max_features='auto'
)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
result = model.score(x_test,y_test)

print(classification_report(y_test, y_pred))
print('acc: ', accuracy_score(y_test, y_pred))
print(result)   # 0.936