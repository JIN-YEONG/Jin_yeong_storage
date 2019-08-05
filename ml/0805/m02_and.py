from sklearn.svm import LinearSVC   # Linear 선형
from sklearn.metrics import accuracy_score   # 정확도(단순히 비교하는 것이기때문에 분류에서만 사용)

# 1. 데이터
# AND 값
x_data = [[0,0], [1,0], [0,1], [1,1]]   
y_data = [0,0,0,1]                    

# 2. 모델
model = LinearSVC()   # 머신러닝 선형모델

'''
SVC (Support Vector Machines)
평면으로는 w값을 구하기 힘든 데이터를 3차원으로 변환하여 w값을 구한다.

LinearSVC()는 선형 모델이다. -> 3차원이 아닌 2차원에서 W값을 구한다.

머신 러닝은 딥러닝에 비해 속도가 빠른 장점이있지만 정확도가 낮다
딥러닝은 모델의 길이를 조절할 수 있기 때문에 정확도가 높다.

빠른 결과를 얻어야 하는 헤커톤 같은 곳에서는 먼저 머신러닝을 사용하여 대략적인 결과를 얻고
머신 러닝을 시작한다.
'''

# 3. 실행
model.fit(x_data, y_data)         

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]  
y_predict = model.predict(x_test)

print(x_test, '의 예측 결과: ', y_predict)
print('acc= ', accuracy_score(y_data, y_predict))   # y_data와 y_predict가 얼마나 일치하는지 단순 비교
                                                    # 회귀 모델에서는 사용 불가