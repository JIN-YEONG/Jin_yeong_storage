# 도마뱀 102페이지


from sklearn.tree import DecisionTreeClassifier
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




tree = DecisionTreeClassifier(max_depth=6, random_state=0, max_features='auto',min_samples_leaf=15)
tree.fit(x_train, y_train)

print('훈련 세트 정확도: {:.3f}'.format(tree.score(x_train,y_train)))
print('테스트 세트 정확도: {:.3f}'.format(tree.score(x_test,y_test)))


print('특성 중요도: \n', tree.feature_importances_)   # 각 컬럼들의 중요도

# 시각화   각 컬럼이 얼마나 중요한지 그래프로 표시
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
훈련 세트 정확도: 0.941
테스트 세트 정확도: 0.937
특성 중요도:
 [0.00000000e+00 5.77672552e-04 0.00000000e+00 5.54326452e-02
 0.00000000e+00 0.00000000e+00 0.00000000e+00 8.34156604e-01
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 3.02455106e-02 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 1.91402508e-02 0.00000000e+00 1.84644195e-04 6.45975229e-03
 0.00000000e+00 0.00000000e+00 0.00000000e+00 5.38029200e-02
 0.00000000e+00 0.00000000e+00]
'''