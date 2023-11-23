# AIStudy

## Machin Learning

<details>
  <summary>
    1. Perceptron Machine Learning
  </summary>
  <div>
    2023-10-03 <a href="https://github.com/ongsiru/AIStudy/blob/main/Linear%20Regression%20Program.ipynb">선형회귀 프로그램</a>
  </div>
</details>

퍼셉트론은 뉴런을 흉내내어 만든 초기 형태의 인공 신경망으로, 다수 입력으로부터 하나의 결과를 내보내는 알고리즘이다. 결국 선형 회귀와 로지스틱 회귀 여러 개를 겹쳐서 사용하면 인공신경망의 딥러닝이 되는 구조다.

<details>
  <summary>
    2. MNIST Dataset Classification
  </summary>
  <div>
    2023-11-07 <a href="https://github.com/ongsiru/AIStudy/blob/main/MNIST%20Dataset%20Classification.ipynb">Classification 프로그램</a>

1.	Layer의 구조
Layer Size의 경우, 레이어 크기를 증가시킨 결과 성능이 향상되었다. 레이어 크기를 늘림으로써 모델은 더 복잡한 패턴을 학습할 수 있다. 이로 인해 정확도가 더 높아진다.
2.	Batch Size
64를 사용한 경우, 정확도가 더 높게 나왔다. 일반적으로 사이즈가 큰 배치는 훈련 속도를 높일 수 있지만, 메모리 요구량이 늘어나고 결과의 유의미한 차이는 없었다.
3.	Optimizer 종류
기울기 최적화 과정에서 SGD를 사용한 경우, Adam보다 정확도가 낮았다. Adam은 모멘텀과 학습률 스케줄링을 자동으로 조절하며 일반적으로 더 좋은 수렴을 제공하는 반면에 SGD는 일정하지 않은 gradient로 파라미터를 업데이트하는 것은 수렴을 방해할 수 있다.
4.	Epoch 수
Epoch 수를 늘리면 모델이 더 많은 훈련을 수행하고 더 높은 정확도를 달성할 수 있다. 그러나 특정 수에 벗어난 Epoch을 사용하면 Overfitting으로 인해 오히려 정확도가 떨어진다. 해당 실험에서 Epoch가 20일 때 이 현상을 발견할 수 있었다.
5.	결론
한정된 학습 데이터에서 Epoch의 수가 커질수록 Overfitting이 발생하고 우리는 중요 파라미터의 값을 변경하거나 여러 가지 피드백과 규제를 부여해 Feature의 영향력을 조절할 수 있다. 
  </div>
</details>

MNIST는 NIST라는 데이터셋을 가공하여 0부터 9까지 인간의 글씨를 28*28 grayscale 이미지로 모아 놓은 것이며 0~255 사이의 숫자 행렬로 표현되어 있다. tensorflow에서 제공하는 딥러닝 라이브러리를 이용하여 MNIST 분류 학습 모델을 실험했다.

<details>
  <summary>
    3. CIFAR Dataset Classification with CNN Model
  </summary>
  <div>
    2023-11-24 <a href="https://github.com/ongsiru/AIStudy/blob/main/CIFAR%20Dataset%20Classification%20with%20CNN%20Model.py">CNN모델 Classification 프로그램</a>
  </div>
</details>

이미지 데이터셋에서 Imbalanced Data와 Balanced Data 간의 성능 차이와 선택된 데이터 클래스의 다양성을 통해 데이터 불균형이 실제 모델 성능에 미치는 영향을 이해할 수 있다. 

## AI Algorithm
