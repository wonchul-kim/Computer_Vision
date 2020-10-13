## Pooling이란

### GeM(Generalized Mean Pooling)

### GAP(Global Average Pooling)
GAP는 기존에 일반적으로 사용되는 구조인 CNN + FC에서 FC를 classifier인 FC를 제거하기 위해 고안된 pooling이다.

이는 CNN의 hidden layer를 거칠수록 high-level의 함축된 정보가 feature에 담기게 되고, 이 feature의 각 평균값을 사용하여 분류하는 것이다.
하지만, FC는 전체의 CNN보다 더 많은 parameter를 가지고 있으므로 계산이 오래 걸리고, 입력단(in_feature)의 크기가 고정되어야 한다는 단점이 있다.
그리고 1차원으로 바뀌기 때문에 위치정보가 모두 소실된다.

즉, GAP를 활용하면 입력되는 크기와 관계없이 사용할 수 있다.

### Max pooling

https://jsideas.net/class_activation_map/