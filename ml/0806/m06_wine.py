# 도마뱀 374페이지
# 파리미터를 수정하여 70% 이상 올리기
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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# 학습하기
model = RandomForestClassifier(n_estimators=238,   # 만들어지는 의사결정나무의 갯수
                               max_depth=50,   # 의사 결정 나무의 깊이
                               max_features='auto',   # 선택할 최대 특성의 수
                               random_state=86   # 랜덤 값
                               # n_job = 2   # 2개의 CPU코어를 병렬적으로 활용
)

''' 
RandomForestClassifier
여러종류의 아웃풋 중 가장 좋은 값이 출력
빠른 속도와 비교적 높은 정확도가 장점이다.

더 높은 정확도를 나타내는 것은 XG-Boost와 딥러닝(keras)가 있다.
딥러닝은 속도가 느리다는 단점이있고 XG-Boost는 환경구축이 어렵고 딥러닝보다 성능이 떨어진다는 단점이다.

의사 결정 트리(decision tree)로 구성되어 있으며 트리에 sample들이 무작위로 들어간다.
여러 의사 결정 트리를 이용하여 결과치를 뽑아내어 가장 좋은 결과를 사용한다.

학습 원리
1. 트레이닝 데이터에서 중복을 허용하여 무작위로 n의 데이터 샘플을 선택 (부트스트랩)
2. 데이터 샘프에서 중복 없이 d개의 데이터 특성값을 선택
3. 의사 결정트리 생성 및 학습
4. 1~3단계 k번 반복
5. 생성된 k개의 의사결정트리를 이용하여 예측된 결과의 평균이나 가장 많이 나온 값을 최종 값으로 결정


RandomForestRegressor
회귀모델
'''

model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)   # 정확도와 같은 값이 나온다.

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))   # 분류 모델 성능 평가
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


둘의 사용법은 비슷하다.

'''