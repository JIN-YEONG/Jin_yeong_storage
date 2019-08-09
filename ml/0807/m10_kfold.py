# sklearn의 모든 분류모델을 실행 한다.


import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators   # sklearn의 모든 모델들이 들어 있다.
import warnings

warnings.filterwarnings('ignore')

iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]

# classifier 알고리즘 모두 추출
warnings.filterwarnings('ignore')
allAlgorithms  = all_estimators(type_filter='classifier')   # sklearn의 모든 분류 모델이 들어 있다.
# allAlgorithms = all_estimators(type_filter='regressor')   # sklearn의 모든 회귀 모델이 들어있다. -> y값이 문자이기 때문에 fit가 불가능

# print(allAlgorithms)
# print(len(allAlgorithms))   # 버전에 따라 개수가 다를 수 있다 
# scikit-learn 0.20.3 -> 31
# scikit=learn 0.21.3 -> 40


# K-분할 클로스 밸리데이션 전용 객체
def allAlgorithmScore(n_split=3, allAlgorithms=allAlgorithms):
	kflod_cv = KFold(n_splits=n_split, shuffle=True)

	score_dic={}
	for (name, algorithm) in allAlgorithms:
	# 각 알고리즘 객체 생성하기
		clf = algorithm()
		# score 메서드를 가진 클래스를 대상으로 하기
		if hasattr(clf, 'score'):   # clf에 score속성이 있는지 판별
			# 크로스 밸리데이션
			scores = cross_val_score(clf, x,y, cv=kflod_cv)   # 교차 검증을 이용한 평가
			# print(name, '의 정답률=',end=' ')
			# print(scores)

			score_dic[name] = scores.mean()

	# print(score_dic)
	sort_dic = sorted(score_dic.items(), key = lambda x: x[1], reverse=True) 

	print('n_split가 ', n_split,'일 때 순위')

	for i in range(3):
		name, value = sort_dic[i]
		print(i+1, end='  ')
		print('%s의 정답률 = %.2f' % (name,value))


for i in range(3,11):
		allAlgorithmScore(i)
		print('=========================================')

# 실습 n_splits를 3~10 까지 입력하여
# 각각 최고 모델 3개 출력