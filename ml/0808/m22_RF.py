# 도마뱀 102페이지


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state= 42
)

# RandomForestClassifier 사용
# 변수에 클래스와 속성값을 지정하고 fit시키면 된다.
tree = RandomForestClassifier(n_estimators=28,max_depth=6, random_state=0)#, max_features='auto',min_samples_leaf=10)
tree.fit(x_train, y_train)

'''
RandomForest 모델
여러개의 의사 결정 트리에 랜덤하게 나눈 값을 넣어서 훈련한다.
n_estimators -> 트리의 개수를 결정 많을수록 좋지만 메모리를 많이 차지하게 된다.

결정 트리를 사용하기 때문에 전처리가 필요없다.
'''


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
훈련 세트 정확도: 0.995
테스트 세트 정확도: 0.965
특성 중요도:
 [0.03028699 0.01818858 0.02062712 0.05667079 0.00857307 0.0026213
 0.03600172 0.10494536 0.00207814 0.00329745 0.02827047 0.00424055
 0.01029455 0.05325126 0.00474    0.00280759 0.00075736 0.0055927
 0.00384898 0.00801273 0.13732529 0.02526349 0.04451167 0.07431954
 0.00677989 0.00779051 0.02935901 0.24952484 0.00901647 0.0110026 ]
'''