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


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(x_train, y_train)

'''
DecisionTree 모델 (의사 결정 트리 모델)
전처리가 필요없다
정보를 1개의 의사결정 트리에 넣어 훈련한다.
max_depth ->트리의 깊이를 조절할수 있다.

기본적으로 결정트리를 사용하는 모델은 속성 값이 비슷하고 전처리 작업이 필요없다.
'''


print('훈련 세트 정확도: {:.3f}'.format(tree.score(x_train,y_train)))
print('테스트 세트 정확도: {:.3f}'.format(tree.score(x_test,y_test)))


print('특성 중요도: \n', tree.feature_importances_)   # 각 컬럼들의 중요도

'''
특성 중요도:
 [0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.01019737 0.04839825
 0.         0.         0.0024156  0.         0.         0.
 0.         0.         0.72682851 0.0458159  0.         0.
 0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
'''