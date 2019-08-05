from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pandas as pd
import numpy as np

iris_data = pd.read_csv('./data/iris.csv',
                        names=['a', 'b', 'c', 'd', 'y'],
                        encoding = 'utf-8')



# 방법 1 LabelEncoder를 배움

# iloc 열의 순서를 이용한 나누기
x= iris_data.iloc[:, 0:4]
y= iris_data.iloc[:, 4]   # y.type = str
# print(x.shape)   # (150,4)
# print(y.shape)   # (150,)

# y열의 값이 문자이기 때문에 숫자로 변경
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y)
y= le.transform(y)   # 값을 종류별로 번호를 붙여 분류한다

'''
#############################################################################

# 방법 2 직접 만듬

# y열의 값이 문자이기 때문에 숫자로 변경
name_mapping = {'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2}
iris_data['y'] = iris_data['y'].map(name_mapping)   # pandas를 이용한 맵핑

x= iris_data.iloc[:, 0:4]
y= iris_data.iloc[:, 4]   # y.type = str

#############################################################################
'''
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, test_size =0.2
)

# print(x_train.shape)   # (120,4)
# print(x_test.shape)   # (30, 4)

# OneHotEncoding
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)


# 모델
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))


model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 실행
model.fit(x_train, y_train, batch_size=1, epochs=100)

# 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

y_predict = model.predict(x_test, batch_size=1)
y_predict = y_predict.round()   # softmax가 0~1값을 반환하기 때문에 반올림하여 0 or 1로 바꾼다.
# print(y_predict)

y_predict = np.argmax(y_predict, axis=1)   # reverse OneHotEncoding 
                                           # np.argmax()   가장 큰 값의 인덱스 반환


# 방법 1
y_predict = le.inverse_transform(y_predict)

'''
# 방법 2
y_pred_value = []
for onhotvalue in y_predict:
    if onhotvalue == 0:
        y_pred_value.append('Iris-setosa')
    elif onhotvalue == 1:
        y_pred_value.append('Iris-versicolor')
    elif onhotvalue == 2:
        y_pred_value.append('Iris-virginica')
    
y_predict =np.array(y_pred_value)
'''

print('acc=', acc)
print(y_predict)
