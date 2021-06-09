import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression  # 머신러닝 라이브러리 중 하나로, 분류, 회귀, 군집화 등의 작업을 할 수 있다.

data = pd.read_csv(r"C:\Users\82105\Downloads\서울특별시 코로나19 확진자 발생동향.csv")    # 컴퓨터 안에 있는 파일경로를 참조함으로서 데이터를 읽어온다.
# 리스트를 사용하면 메모리를 많이 차지함으로 numpy를 사용
x = np.array([])       # 초기화
y = np.array([])       # 초기화

# date = 2020/04/22 을 1로 잡고 2021/06/08 까지 오름차순 숫자 목록
for i in data['date']:      # date열 참조
    x = np.append(x, np.array([i]))     # date 요소들을 x 배열에 추가
# plus = date기준으로 일일 추가 확진자 수
for j in data['plus']:      # plus열 참조
    y = np.append(y, np.array([j]))     # plus 요소들을 x 배열에 추가

lr = LinearRegression()     # 변수 초기화

x = x.reshape(-1,1)     # x를 다차원으로 바꿔주는 코드 -1은 남은 배열값을 자동 처리해주는 값 1은 열의 값

lr.fit(x, y)       # lr 에 x,y 데이터를 준다.

y_pred = lr.predict(x)      # predict는 데이터를 예측해주는 함수 , 예측 결과를 y_pred에 저장

print("기울기 :", lr.coef_[0])        # 기울기
print("y절편 :", lr.intercept_)        # y절편
print("x가 200 일 때의 y 값 :", lr.predict([[200]])[0])     # predit 함수에 200을 넣어줌으로 y값이 나온다 그냥 출력하면 []로 표시해서 [0]을 뒤에 넣었다.
plt.scatter(x, y)     # 산점도 그래프 그려주는 코드(점을 뿌려준다 생각)
plt.plot(x, y_pred, color='red')    # 선을 그려준다 , 선 색 지정 가능
plt.show()      # 실제 표시
