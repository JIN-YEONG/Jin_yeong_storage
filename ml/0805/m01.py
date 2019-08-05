# 싸인 값의 시각화

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10,0.1)   # 0~10 값 0.1씩증가
y = np.sin(x)   # 싸인함수

plt.plot(x,y)
plt.show()

