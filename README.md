# HyperparameterTuning
Comparing the performance between dynamic_grid_search, dynamic_random_search, and bayesian_optimization

<br>
<br>


머신러닝에서 하이퍼파라미터를 최적화할 때 일반적으로 격자탐색보다 임의탐색이 더 효율적이라고 한다.  

그러나 다음 방법대로 하면 격자탐색이 더 나을 수도 있지 않나 하는 의문이 들었다.  

1. 매우 넓은 간격으로 격자탐색을 시행한다.  
2. '1.' 에서의 최적점 근처에서 좀 더 좁은 범위로 격자탐색을 시행한다.  
3. 이전 단계에서의 최적점 근처에서 좀 더 좁은 범위로 격자탐색을 시행하는 것을 반복한다. 충분히 여러 번 반복하거나 일정 기준을 충족하면 종료한다.  

이러면 마치 Binary Search나 Greedy Algorithm처럼 동적으로 탐색 영역을 줄여가며 정교하게 하이퍼파라미터를 도출할 수 있다.  
그러나 임의탐색 역시 비슷한 방식으로 동적으로 시행할 수 있다.  
그리고 이미 하이퍼파라미터 탐색 방법으로 널리 쓰이는 Bayesian Optimization도 있다. 괜히 널리 쓰이는 게 아닐 것이다.  

이에 동적 격자탐색과 동적 임의탐색, Bayesian Optimization의 모델 성능을 비교해 보기 위해 만든 코드다.  

---

동적 하이퍼파라미터 탐색은 다음과 같이 진행한다.  

1. 모델의 표준에서 제시하는 추천 하이퍼파라미터의 주변 공간을 전체 탐색 대상 공간으로 한다.  
2. 탐색 대상 공간에서 1차적으로 탐색하여 최적 지점을 찾는다. 탐색 결과 계산에는 Validation set을 사용한다.  
3. 2. 에서 찾은 최적 지점 근처로 탐색 대상 공간을 줄인 후, 2차적으로 탐색하여 최적 지점을 개선한다. 이 과정을 유의미한 성능 향상이 없거나 일정 횟수에 도달할 때까지 반복한다.  
4. 반복 종료 후 Test set으로 최종 모델의 성능을 도출한다.  

이때 2.와 3.에서 사용하는 탐색 방법은 다음과 같다.  

동적 격자탐색의 경우,  
1. 탐색 대상 공간에 3*3 격자를 만든 후, 격자선들의 왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래 교차점과 중심에 있는 점, 총 5개 점을 잡는다.  
2. 위 5개 점에 위치한 하이퍼파라미터를 대입한 5개 모델을 만들어 훈련시켜 성능을 도출한다.  
2-1. 이때 초기 탐색 대상 공간에서 처음으로 탐색한다면, 상위 2개 모델을 뽑는다. 상위 2개 모델의 성능 차이가 너무 크면 최상위 모델만 선택하여 다음 탐색으로 넘긴다. 그렇지 않으면 위 2개 하이퍼파라미터 조합에 대해 각각 다음 탐색을 시행한다.  
2-2. 탐색을 2번 이상 반복했다면 상위 1개 모델을 뽑는다. 이때 뽑은 모델의 성능이 이전 탐색에서 뽑힌 모델의 성능보다 유의미하게 개선되지 않았거나, 탐색 횟수가 미리 설정한 횟수에 도달하였다면 탐색을 종료한다. 이전 모델의 성능이 더 좋았다면 이전 모델을 반환하고, 그렇지 않다면 현재 모델을 반환한다. 탐색 종료 조건이 충족되지 않으면 다음 탐색을 시행한다.  

2. 단계에서, 뽑힌 모델을 따로 저장해 놓는다. 그 모델이 필요해지지 않으면 삭제한다. 예를 들면 다음과 같다.  
iteration 1: 선정 모델 저장  
iteration 2: 선정 모델 저장, iteration 1과 비교  
iteration 3: 선정 모델 저장, iteration 2와 비교. iteration 1에서 저장한 모델 삭제  
iteration 4: 선정 모델 저장, iteration 3과 비교. iteration 2에서 저장한 모델 삭제  

임의 격자탐색의 경우,  
1. 탐색 대상 공간에서 uniform하게 5개 점을 랜덤추출한다.  
2. 부터는 동적 격자탐색의 방법과 같다. 단, 2-1에서 최상위 모델만 선택하는 조건에 상위 2개 지점이 너무 가까울 때를 추가한다.  

Bayesian Optimization은 Gaussian Process 기반 방법을 따른다. 획득함수는 ```qLogNEI```를 사용한다.  

---

이상의 3가지 방법을  
[dynamic_grid_search.py](https://github.com/SeizeMyDay/HyperparameterTuning/blob/main/dynamic_grid_search.py)  
[dynamic_random_search.py](https://github.com/SeizeMyDay/HyperparameterTuning/blob/main/dynamic_random_search.py)  
[bayesian_optimization.py](https://github.com/SeizeMyDay/HyperparameterTuning/blob/main/bayesian_optimization.py)  
에 구현했고,  

[run_py](https://github.com/SeizeMyDay/HyperparameterTuning/blob/main/run.py)  
로 실행하여 성능을 비교할 수 있도록 했다.  

하이퍼파라미터 튜닝의 대상이 되는 모델은 CNN 모델이며, 세부 구조는 다음과 같다.  
```
- Conv 32 → ReLU → Pool → Conv 64 → ReLU → Pool → FC 128 → Output 10
- Optimizer: AdamW
- Default hyperparams: learning rate \(10^{-3}\), weight decay \(10^{-4}\)
- training condition: 20 epoch, batch size 128
```
훈련 및 평가 데이터는 ```FashionMNIST```이고, train 50,000 / validation 10,000 으로 분할하여 사용한다.  

이후 탐색 코드에서는 다음 코드로 후보 하나를 평가한다.  
```
from base_model import (
    DataConfig, TrainConfig, HyperParameters,
    prepare_data, train_candidate, save_result,
)

data = prepare_data(DataConfig())  # 탐색 시작 전에 한 번만 실행
config = TrainConfig(epochs=20)

result = train_candidate(
    data=data,
    hyperparameters=HyperParameters(
        learning_rate=1e-3,
        weight_decay=1e-4,
    ),
    config=config,
    seed=42,
)

score = result.val_accuracy       # 최대화할 값, 0~1
save_result(result, "runs/candidate_001")
```

---

ChatGPT가 예상한 실행 소요시간이 13시간 가량이나 되어 아직은 돌릴 엄두를 내지 못했다..  
조만간 구글코랩으로 돌려 볼 생각이다.  
아니면 결과가 시원찮아질 수 있지만 임시방편으로 데이터를 줄여서 돌리는 것도 생각 중이다.
