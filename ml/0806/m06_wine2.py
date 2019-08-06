# wine데이터의 분포

import matplotlib.pyplot as plt
import pandas as pd

# 와인 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')

# 품질 데이터별로 그룹을 나누고 수 새어보기
count_data = wine.groupby('quality')['quality'].count()
# 데이터프레임 wine에서 'quality'열의 값으로 그룹화 하여 'quality'열의 값별 개수를 반환
print(count_data)

count_data.plot()   # 시각화
plt.savefig('wine-count-plt.png')   # 시각화 저장
plt.show()   # 시각화 출력