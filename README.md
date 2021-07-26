# Finals

## 1) SimulatedAnnealing

### 코드 구현

```python
def fit(x):
    return 0.16*x*x*x*x-x*x+0.37*x+5  # 4차 함수(예시)
def isNeighborBetter(f0 , f1): 
    return f0>f1        # f0과 f1비교
```

fit의 return 값을 본인이 원하는 함수로 바꾸면 된다.  
일단 코드설명에서는 교수님께서 예로 드신 4차 함수로 하겠습니다.

#### SimulatedAnnealing 함수  
```python
def solve(t,lower,upper,a,niter,hist):
    r0 = random.random()       # 랜덤 (0.0 ~ 1.0 사이 값)
    x0 = r0*(upper-lower)+lower  # 후보 해의 값을 임의로 구하기 위해 random값에 연산을 해준다.
    f0 = fit(x0) # 후보 해
    hist.append(f0)     # parameter로 받아온 배열에 각 해들을 넣어준다.
    for i in range(niter):  # niter = 반복 횟수
        kt = int(2/t)       # kt는 t가 증가할수록 감소해야함으로 2/t로 해두었다.
        for j in range(kt):     # kt만큼 반복
            r1 = random.random()
            x1 = r1*(upper - lower) + lower   # 이웃 해의 값을 임의로 구하기 위해 random값에 연산을 해준다.
            f1 = fit(x1)  # 이웃 해
            if isNeighborBetter(f0,f1):     # 비교해서 후보 해가 더 높으면 
                x0 = x1
                f0 = f1     # 후보 해의 값을 이웃 해로 교체
                hist.append(f0)     # 배열에 추가
            else:       # 비교해서 이웃 해가 더 높으면
                d = abs(f1 - f0)        # f1과 f0의 차를 절댓값으로 감싼 d
                p0 = math.exp(-d/t)     #   p0 을 구해준다 p0 = 자유롭게 탐색할 확률, t와 p0는 정비례 해야하고 d와는 반비례 해야한다.
                if r1<p0:       # 만약 p0이 random값보다 크면 교환 할 기회를 준다.
                    x0 = x1     
                    f0 = f1     # 교환
                    hist.append(f0)     #배열 추가
        t *= a      # 반복할 때 마다 t값에 a값 곱해주기
    return f0       # f0값 return
```
먼저 후보 해의 값을 임의로 구하기 위해 random 값에 연산을 해서 x0에 저장을 한 다음 앞에서 정의한 fit함수로 f0값을 구해준다.  
이제 niter(반복 횟수) 만큼 반복문을 돌면서 그 안에 다시 kt만큼 반복하는데 여기서 kt가 t와 반비례 해야한다.  
이제 이웃 해의 값을 임의로 구하기위해 위에서 했듯이 random 값에 연산을 해서 x1에 저장하고 fit함수로 f1값을 구해준다.  
이제 앞서 정의한 isNeighborBetter함수로 f0값과 f1값을 비교해서 후보 해가 더 높으면 후보 해의 값을 이웃 해로 교체해준다.  
추가로 hist에 삽입해주고 만약 이웃해가 더 높으면 f1과 f0의 차를 구해서 이를 이용해 p0를 구해준다.  
만약 p0가 random 값보다 크면 후보 해의 값을 교환 할 기회를준다. 마지막으로 다시 hist배열에 f0값을 삽입한다.  
반복문이 끝날 때마다 t값에 a만큼 곱해줘서 반복 할 때마다 최적값에 가깝게 해준다.  
마지막으로 return f0!  

---

```main
hist = []   # 함수에 넘겨줄 배열 생성
print("전역 최적점 :",solve(1,-100,100,0.95,100,hist))
print("추론 과정 :",hist)
```
hist라는 함수에 넘겨줄 배열을 생성하고  
print함수로 f0값과 함수 호출 후 값이 추가된 hist 배열을 출력한다.  

---
### 결과값
- 출력  
![image](https://user-images.githubusercontent.com/80373033/121330579-19f5b880-c951-11eb-8719-df6dc6c1ebc9.png)

- (추론 과정값(hist의 각 요소)들을 그래프로 표현  
![image](https://user-images.githubusercontent.com/80373033/121330650-2aa62e80-c951-11eb-8305-08acc9221ddc.png)  
결국엔 코드가 실행되면서 반복될 떄마다(시간이 지남에 따라) 결국에 최적값에 수렴하는 결과를 보여준다.  

### 결과값과 함수비교  
위 코드에 사용한 함수를 실제 그래프로 표현한 것이다.  
![image](https://user-images.githubusercontent.com/80373033/121330755-43164900-c951-11eb-8961-f4f284290b1e.png)  
사진에 나온 최솟값과 추론해서 나온 최적값이 매우 유사해서 잘 나왔다고 할 수 있다.

### 다른 함수 넣어보기
```python
def fit(x):
    return 4*x*x*x*x + 5*x*x - 7 
```
보다시피  
![image](https://user-images.githubusercontent.com/80373033/121338716-bb343d00-c958-11eb-93a3-9e930b11379f.png)  
  
![image](https://user-images.githubusercontent.com/80373033/121338797-d30bc100-c958-11eb-8f24-e4e8969dd680.png)  
최적값과 최솟값이 매우 유사하다. 결국 다른 함수를 넣어도 예상한 결과와 같이 나오는 것을 알 수 있다.

## 2-1) 공공 데이터와 모의 담금질 기법을 사용해 최적값 찾기
데이터는 전국 일일 코로나 추가 확진자 수를 가져왔다.  
[서울 열린 데이터 광장](http://data.seoul.go.kr/dataList/OA-20461/S/1/datasetView.do#AXexec)  
이곳에서 서울은 물론 전국 확진자 수를 가져올 수 있다.

### 코드 설명
```python
import csv      # csv 파일을 읽기 위한 라이브러리
import random  
import math

def isNeighborBetter(f0 , f1): 
    return f0<f1        # f0과 f1비교
```

먼저 csv 파일을 읽기위해 import csv를 해준다음 random과 math또한 import 해준다.  
isNeighborBetter은 앞선 1번 코드에서와 같지만 여기서 최적값이 최댓값으로하면 f1>f0을 하고 최솟값으로 하면 f1<f0으로 해주면 된다.

---


```python
def solve(t,a,niter,hist,lst):
    f0 = random.choice(lst)        # parameter에서 받아온 lst중에서 랜덤으로 값 하나를 f0에 저장
    hist.append(f0)     # parameter로 받아온 배열에 각 해들을 넣어준다.
    for i in range(niter):  # niter = 반복 횟수
        kt = int(2/t)       # kt는 t가 증가할수록 감소해야함으로 2/t로 해두었다.
        for j in range(kt):     # kt만큼 반복
            f1 = random.choice(lst)       # parameter에서 받아온 lst중에서 랜덤으로 값 하나를 f1에 저장
            if isNeighborBetter(f0,f1):     # 비교해서 후보 해가 더 높으면 
                f0 = f1     # 이웃해와 후보해 교환
                hist.append(f0)     # 배열에 추가
            else:       # 비교해서 이웃 해가 더 높으면
                d = abs(f1 - f0)        # f1과 f0의 차를 절댓값으로 감싼 d
                p0 = math.exp(-d/t)     #   p0 을 구해준다 p0 = 자유롭게 탐색할 확률, t와 p0는 정비례 해야하고 d와는 반비례 해야한다.
                if random.random()<p0:       # 만약 p0이 random값보다 크면 교환 할 기회를 준다.
                    f0 = f1     # 교환
                    hist.append(f0)     #배열 추가
        t *= a      # 반복할 때 마다 t값에 a값 곱해주기
    return f0       # f0값 return
```

1번 코드와 바뀐점은 f0, f1값을 그저 random을 사용하여 임의로 정수값을 구한다음 fit함수를 적용해서 구했지만  
이번엔 데이터를 사용함으로 데이터들을 배열로 가져온 인자값인 lst에서 random.choice를 이용해 배열내에서 랜덤으로 하나를 가져왔다.  
그렇게 f0값과 f1값을 구하는 방식만 바꿔주면 된다.  

```python
f = open('C:/Users/82105/Downloads/전국 코로나19 확진자 발생동향.csv', 'r', encoding='utf-8')     # 컴퓨터 안에 있는 csv파일을 f에 저장
rdr = csv.reader(f)     # f를 읽어서 rdr에 저장

lst = []        # 배열 초기화
for line in rdr:       # rdr에서 각 열들을 line에 넣고 반복
    lst.append(int(line[0]))    #  line[0] 로 첫번 째 열의 각 요소들만 lst이 추가
f.close()    # f 닫아주기

hist = []       # 함수에 parameter로 넘겨줄 배열 초기화
print("일일 코로나19 확진자가 가장 많았을 때의 확진자 수는?",solve(1,0.95,100,hist,lst))     # 출력
```
먼저 컴퓨터 안에 저장한 csv파일을 읽어서 rdr에 저장시켜준다.  
그런 다음 rdr에서 첫 번쨰 열의 값들(일일 확진자 수)만 lst에 추가시켜준다. 그런 다음 solve함수에 lst도 같이 넣어주고 실행  

---
### 출력값  
- 일일 확진자가 가장 많았을 때의 확진자 수를 최적값으로  
![image](https://user-images.githubusercontent.com/80373033/121359961-f6416b00-c96e-11eb-96d4-41bc9c7ce8ca.png)  
보다시피 최댓값을 기준으로 최적값을 구했기 때문에 데이터 중에 가장 큰 값을 표현했다.  
- 일일 확진자가 가장 적었을 때의 확진자 수를 최적값으로  
![image](https://user-images.githubusercontent.com/80373033/121360036-05281d80-c96f-11eb-8bc9-264cf8090c41.png)  
반대로 일일 확진자가 가장 적게 나왔을 때도 코드 일부분만 수정하면 가능하다.  
이렇게 공공 데이터를 가지고 표현했다.  
2-2 에선 선형 회귀를 사용해 데이터 추론을 해보았다.  

## 2-2) Regression
### 공공데이터를 가져와서 선형 회귀로 데이터를 추론
데이터는 전국 일일 코로나 추가 확진자 수를 가져왔다.  
[서울 열린 데이터 광장](http://data.seoul.go.kr/dataList/OA-20461/S/1/datasetView.do#AXexec)  
이곳에서 서울은 물론 전국 확진자 수를 가져올 수 있다.

### 코드 설명
평소 알고리즘을 공부할 때는 vscode를 사용하지만 이번 과제는 pycharm에서 수행하는 것이 간편하여 pycharm에서 수행하였다.
```python
import numpy as np      # 다차원 배열을 위한 라이브러리
import matplotlib.pyplot as plt     # 데이터를 차트나 플롯 형태로 그려주는 라이브러리
import pandas as pd     # 데이터를 읽기 위한 라이브러리
from sklearn.linear_model import LinearRegression       # 분류, 회귀, 군집화 등의 작업을 할 수 있는 라이브러리
```
먼저 import하는 구문이다 물론 4가지 모두 외부 라이브러리이기에 pip install을 해주어야 한다.  
먼저 numpy는 과학 계산용 라이브러리로 데이터를 처리함으로 다차원 배열을 처리하는데 필요하다.  
matplotlib는 데이터를 차트나 플롯 형태로 그려주는 라이브러리이다.  
pandas는 데이터를 읽기 위한 라이브러리이다.  
sklearn은 파이썬 라이브러리중 머신러닝에 가장 유명하며 분류, 회귀, 군집화 등의 작업을 할 수 있다.  
그중 LinearRegression을 사용하여 선형 회귀 작업을 한다.  
```python
data = pd.read_csv(r"C:\Users\82105\Downloads\전국 코로나19 확진자 발생동향.csv")

x = np.array([])
y = np.array([])

# date = 2020/04/22 을 1로 잡고 2021/06/08 까지 오름차순 숫자 목록
for i in data['date']:      # date열 참조
    x = np.append(x, np.array([i]))     # date 요소들을 x 배열에 추가
# plus = date기준으로 일일 추가 확진자 수
for j in data['plus']:      # plus열 참조
    y = np.append(y, np.array([j]))     # plus 요소들을 x 배열에 추가

```
이 부분은 먼저 pandas를 이용해 내가 다운받은 csv파일(경로 참조)을 읽어서 data에 넣어준다  
date와 plus는 각각 주석에서 알 수 있듯 날짜를 오름차순으로 한 숫자 이며 plus는 일일 추가 확진자 수이다.  
그런다음 x와 y를 numpy array로 초기화해주고 data에 coulum의 영역을 반복문으로 참조하면 coulum의 요소들이 하나씩 i에 대입되어진다.  
그것들을 각 x, y에 넣어준다.
```python 
lr = LinearRegression()   # 변수 초기화

x = x.reshape(-1,1)     # x를 다차원으로 바꿔주는 코드 -1은 남은 배열값을 자동 처리해주는 값 1은 열의 값

lr.fit(x, y)       # lr 에 x,y 데이터를 준다.

y_pred = lr.predict(x)      # predict는 데이터를 예측해주는 함수 , 예측 결과를 y_pred에 저장

print("기울기 :", lr.coef_[0])        # 기울기
print("절편 :", lr.intercept_)        # 절편
print("x가 200 일 때의 y 값 :", lr.predict([[200]])[0])     # predit 함수에 200을 넣어줌으로 y값이 나온다 그냥 출력하면 []로 표시해서 [0]을 뒤에 넣었다.

plt.scatter(x, y)     # 산점도 그래프 그려주는 코드(점을 뿌려준다 생각)
plt.plot(x, y_pred, color='red')    # 선을 그려준다 , 선 색 지정 가능
plt.show()      # 실제 그래프 표현
```
lr이란 변수를 초기화 시켜준다. 먼저 데이터를 넣은 x배열을 다차원으로 바꿔준다. 그런 다음 fit함수를 통해 lr에 x와 y 데이터를 넣어준다.  
그리고 predict를 함수를 사용해 데이터를 예측해서 y_pred에 저장시킨다.  
coef_와 intercept함수로 기울기와 절편을 구하고 추가로 x 값에 맞는 y값도 구해서 print한 다음에 plt.scatter로 산점도 그래프를 그려준다. 그저 데이터들을 x와 y에 맞게 점을 뿌려준다 생각하면 편하다.  
plot 함수에 x와 예측한 데이터인 y_pred를 넣어주어 선을 그리게 된다. color는 red로 하였다.  
마지막으로 show 함수를 사용해서 실행함으로 그래프를 표현해준다
### 결과값
![image](https://user-images.githubusercontent.com/80373033/121339080-15cd9900-c959-11eb-91dd-8bee5f535897.png)  
- 출력값  
![image](https://user-images.githubusercontent.com/80373033/121339119-2120c480-c959-11eb-9c87-c65bf16d94e8.png)  
이렇게 되면 그래프에 나오는 예측 선(빨간 선)의 y = ax+b의 a와 b값을 구할 수 있다. 그래서 결과는 y = 1.805x - 47.67 이다.
- y = 1.805x - 47.67 를 그래프로 표현  
 ![image](https://user-images.githubusercontent.com/80373033/121339186-30a00d80-c959-11eb-9a57-f1396ccd4c0c.png)  
다르게 보일 수 있지만
위의 출력값에서 X가 200일 떄의 값과 함수 그래프에서 표시한 X가 200일 떄의 값이 동일하다는 것으로 사실은 같은 함수를 가지는 선이란 것을 알 수 있습니다.


##### 한 학기동안 수고 많으셨습니다! 교수님, 많은 도움 되었습니다.

