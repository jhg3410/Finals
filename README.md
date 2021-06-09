# Finals

## 1) SimulatedAnnealing

### 코드 구현

```python
def fit(x):
    return 0.16*x*x*x*x-x*x+0.37*x+5  # 4차 함수(예시)
def isNeighborBetter(f0 , f1): 
    return f0>f1        # f0과 f1비교
```

