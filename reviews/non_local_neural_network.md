# Non-local Neural Network

### Abstract
기존의 CNN과 RNN은 학습하는 데에 있어서 한 번의 계산(처리)을 하는 영역이 local negiborhood(time and space 관점에서)라는 단점이 있다. 예를 들어, 32x32 이미지를 처리하는 데에 있어서
3x3의 kernel을 사용한다면, 한 번에 처리할 수 있는 영역 또한 3x3으로서 제한이 된다고 할 수 있다. 본 논문은 이 단점을 해결하고자, 2004년에 발표된 [non-local operation means](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf)이론을 발전시켜 이를 CNN에 응용하였다. 논문에서는 다음과 같이 설명하고 있다. 

> Our non-local operation computes the reponse at a position as a weighted sum of the features at all positions.

단순히 해석하자면, 어떠한 영역을 처리하는 데에 있어서 전체에 대한 feature들의 weighted sum으로서 계산을 한다는 것이다. (나중에 연산과정을 보면 직관적으로 이해감)

그리고 