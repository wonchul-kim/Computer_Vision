# Non-local Neural Network

### Abstract
기존의 CNN과 RNN은 학습하는 데에 있어서 한 번의 계산(처리)을 하는 영역이 local negiborhood(time and space 관점에서)라는 단점이 있다. 예를 들어, 32x32 이미지를 처리하는 데에 있어서
3x3의 kernel을 사용한다면, 한 번에 처리할 수 있는 영역 또한 3x3으로서 제한이 된다고 할 수 있다. 본 논문은 이 단점을 해결하고자, 2004년에 발표된 [non-local operation means](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf)이론을 발전시켜 이를 CNN에 응용하였다. 논문에서는 다음과 같이 설명하고 있다. 

> Our non-local operation computes the reponse at a position as a weighted sum of the features at all positions.

단순히 해석하자면, 어떠한 영역을 처리하는 데에 있어서 전체에 대한 feature들의 weighted sum으로서 계산을 한다는 것이다. (나중에 연산과정을 보면 직관적으로 이해감)

### Intro.
CNN의 연산과정은 여러 개의 kernel을 사용하여 주어진 이미지의 feature를 모아가는 방식이라고 할 수 있다. 하지만, 이 kernel들의 크기가 제한되어 있다는 점에서 이미지를 전체적으로 분석하는 것이 아닌 지역적(local)으로 feature를 추출한다. 이는 receptive field가 제한적이고 작다고 할 수 있으며, 이러한 단점을 극복하기 위해서 여러 층으로 구성을 하지만 효율적이지는 않다. 본 논문은 이러한 local operator의 단점을 보완해주는 non-local operator를 제안하였고, 이는 non-local means filter에서 영감을 받았다고 한다. 

* Non-local Means Filter

노이즈를 제거하기 위한 filter로서, 한 장의 이미지 내에서 노이즈를 제거하고자하는 영역과 비슷한 영역들을 이용하여 노이즈를 제거하는 것이다. 그리고 이 때 이미지의 전체 영역을 모두 활용한다는 점에서 non-local이라고 제시하는 듯 하다.

### Non-local Neural Networks

Non-local neural network는 기존의 CNN을 여러 층으로 구성하는 모델의 구조에서 삽입될 수 있다는 점에서 더욱 용이하며, 다음과 같은 수식으로 나타난다.

$$ y_i = \frac{1}{C(x)}\sum_{\forall j} f(x_i, x_j)g(x_j) $$

