import csv
import random
import math


def isNeighborBetter(f0 , f1): 
    return f0>f1        # f0과 f1비교

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

f = open('C:/Users/82105/Downloads/서울특별시 코로나19 확진자 발생동향.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

lst = []
for line in rdr:
    lst.append(int(line[0]))
f.close()    


hist = []
print("일일 코로나19 확진자가 가장 적었을 때의 확진자 수는?",solve(1,0.95,100,hist,lst))
print("")