import random
import math

def fit(x):
    return 4*x*x*x*x + 5*x*x - 7  # 4차 함수(예시)
def isNeighborBetter(f0 , f1): 
    return f0>f1        # f0과 f1비교

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
                f0 = f1     # 이웃해와 후보해 교환
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

hist = []   # 함수에 넘겨줄 배열 생성
print("전역 최적점 :",solve(1,-100,100,0.95,100,hist))
print("추론 과정 :",hist)