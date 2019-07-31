from keras.models import Sequential

fiter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), padding = 'same',input_shape=(5,5,1)))    # fiter_size->output, 자른 만든 데이터를 32장을 만들어라
                                                                     # kernel_size -> 하나의 이미지를 어떤 크기로 자를지 결정
                                                                     # input_shape -> input이미지의 크기, 흑백=1, 컬러=3
                                                                     # padding -> valid/same  valid->유지x(기본), same->유지o
                                                                     #                        원본 이미지의 크기를 유지하기 위해 사용
                                                                     #                        출력 데이터의 크기가 줄어드는 것을 방지
                                                                     #                        원본 이미지에서부터 페딩을 입힌다.(의미없는 값 추가)
model.add(Conv2D(16, (2,2)))
model.add(MaxPooling2D(2,2))   # 입력을 커널이 겹치치 않게 잘라 커널내의 가장 큰 특성값을 반환
                               # 자르고 남는 값은 버려진다. 
model.add(Conv2D(8, (2,2)))

model.add(Flatten())   # 2D데이터를 1줄로 핀다.

model.add(Dense(1))   # Dense모델 

model.summary()
# (2,2)로 나누면 16장 즉 4x4개가 나옴
# output=7은 4x4를 7장 만들라는 이야기(4x4x7)
# param의 개수 -> output* kernel +1
#              -> 7*(2*2+1)

# mnist
# 0~9 손글씨 

# cnn
# 이미지 데이터를 1줄로 만들어서 쌓는다.
