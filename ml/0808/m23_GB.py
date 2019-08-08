# 도마뱀 101페이지 ~ 125페이지


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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

GradientBoosting 모델
전처리 불필요
여러개의 의사 결정 트리를 만들어 사용하는 모델 
이전 트리의 오차를 보완하는 방식으로 새로운 트리를 만든다.
learning_rage -> 이전 트리의 오차를 얼마나 강하게 보정할 것인지 결정

'''


tree = GradientBoostingClassifier(max_depth=6,random_state=0, learning_rate=0.01)
tree.fit(x_train, y_train)

print('훈련 세트 정확도: {:.3f}'.format(tree.score(x_train,y_train)))
print('테스트 세트 정확도: {:.3f}'.format(tree.score(x_test,y_test)))

# n_estimators : 클수록 좋다, 단점: 메모리 많이 차지, 기본값:100
# n_jobs = -1 : cpu 병렬처리
# max_features : 기본값 써라('auto')

print('특성 중요도: \n', tree.feature_importances_)   # 각 컬럼들의 중요도

# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('특성의 중요도')
    plt.ylabel('특성')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
plt.show()

'''
훈련 세트 정확도: 0.998
테스트 세트 정확도: 0.937
특성 중요도:
 [0.00073145 0.00795859 0.00102671 0.00055057 0.00292969 0.00083718
 0.00046836 0.01894033 0.00233953 0.         0.00165472 0.02197078
 0.00134846 0.01147874 0.00572628 0.0047823  0.00169273 0.00047459
 0.00122461 0.00371535 0.3220678  0.04333489 0.38389402 0.00653795
 0.00393077 0.00102832 0.01016174 0.12845926 0.00945848 0.0012758 ]
'''