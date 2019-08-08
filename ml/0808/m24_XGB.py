# 도마뱀 101페이지 ~ 125페이지


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state= 42
)

# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(x_train, y_train)

# print('훈련 세트 정확도: {:.3f}'.format(tree.score(x_train,y_train)))
# print('테스트 세트 정확도: {:.3f}'.format(tree.score(x_test,y_test)))

'''
도마뱀 172페이지 참고

DecisionTree 모델 (의사 결정 트리 모델)
전처리가 필요없다
정보를 1개의 의사결정 트리에 넣어 훈련한다.

속도가빠르고 시각화하기 좋고 설명이 쉬움
max_depth ->트리의 깊이를 조절할수 있다.

RandomForest 모델
여러개의 의사 결정 트리에 랜덤하게 나눈 값을 넣어서 훈련한다.
DecisionTree모델보다 거의 항상 좋은 성능, 매우 안전적이고 강력함, 
고차원 희소데이터에는 잘 안 맞음
n_estimators -> 트리의 개수를 결정 많을수록 좋지만 메모리를 많이 차지하게 된다.

GradientBoosting 모델
여러개의 의사 결정 트리에 사용하는 모델 
이전 트리의 오차를 보완하는 방식으로 트리를 만든다.
RandomForest 모델 보다 조금 더 성능이 좋음, RandomForest 모델보다 학습은 느리나
예측은 빠르고 메모리를 조금 사용한다. 매개변수 튜닝이 많이 필요하다.
learning_rage -> 이전 트리의 오차를 얼마나 강하게 보정할 것인지 결정

XGBoost 모델
GradientBoosting모델의 성능과 속도를 개선한 모델
가장 정확도가 높은 머신러닝 기법이다.


위 4개의 모델은 모두 의사결정 트리를 사용하는 모델로서 
전처리가 필요없고 높은 정확도를 보장한다.
범위 밖의 데이터를 맞출수 없다.(사실 딥러닝도 똑같다.)
train에 대한 과적합이 발생한다는 단점이 있다.
'''


tree = XGBClassifier(n_estimators=300,max_depth=3,n_jobs=-1)
# earlystopping 가능
tree.fit(x_train, y_train)

print('훈련 세트 정확도: {:.3f}'.format(tree.score(x_train,y_train)))
print('테스트 세트 정확도: {:.3f}'.format(tree.score(x_test,y_test)))

# n_estimators : 클수록 좋다, 단점: 메모리 많이 차지, 기본값:100
# n_jobs = -1 : cpu 병렬처리
# max_features : 기본값 써라('auto')

print('특성 중요도: \n', tree.feature_importances_)   # 각 컬럼들의 중요도

# # 시각화
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel('특성의 중요도')
#     plt.ylabel('특성')
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(tree)
# plt.show()

'''
훈련 세트 정확도: 1.000
테스트 세트 정확도: 0.965
특성 중요도:
 [0.         0.01358085 0.         0.0161722  0.00473046 0.0036419
 0.00205341 0.02367493 0.00256584 0.00784692 0.01305037 0.0592582
 0.01126808 0.00653429 0.00390979 0.00194395 0.01776742 0.00144122
 0.00200397 0.00733425 0.45877483 0.01481981 0.15572244 0.031817
 0.01297333 0.01420707 0.01341462 0.09359571 0.00313983 0.0027573 ]
'''