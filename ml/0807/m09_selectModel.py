# sklearn의 모든 분류모델을 실행 한다.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators   # sklearn의 모든 모델들이 들어 있다.
import warnings

warnings.filterwarnings('ignore')   # 오류 메세지 무시
# 여러 모델들을 무작성 사용하기 때문에 여러 오류 메세지가 출력된다.

iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]


# 학습 전용과 테스트 정용 분리
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8)

# classifier 알고리즘 모두 추출 
warnings.filterwarnings('ignore')
allAlgorithms  = all_estimators(type_filter='classifier')   # sklearn의 모든 분류 모델이 들어 있다.
# allAlgorithms = all_estimators(type_filter='regressor')   # sklearn의 모든 회귀 모델이 들어있다. -> y값이 문자이기 때문에 fit가 불가능

print(allAlgorithms)
print(len(allAlgorithms))   # 버전에 따라 개수가 다를 수 있다 
# scikit-learn 0.20.3 -> 31
# scikit=learn 0.21.3 -> 40


# 모든 모델을 fit
for (name, algorithm) in allAlgorithms:   # all_estimators의 반환값은 (name, algorithm)으로 구성
    # 각 알고리즘 객체 생성하기
    clf = algorithm()

    # 학습하고 평가
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, '의 정답률 =', accuracy_score(y_test, y_pred))
