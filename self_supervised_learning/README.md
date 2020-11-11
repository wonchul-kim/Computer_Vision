# Self-supervised Learning

## Motivation & Definition

Supervised learning은 label이 모두 되어있는 데이터를 기반으로 하는 지도학습을 의미하며, 이미 인간의 능력보다 더 뛰어나다는 평가를 받을정도로 성능이 개선되었다. 그리고 많은 모델들과 알고리즘들이 개발되었고, ImageNet과 더불어 다양한 오픈 소스의 데이터들도 공개가 되고 있다. 하지만, 여전히 label이 되어 있는 양질의 데이터가 필요하다는 점에서 이를 해결하기 위한 시도들이 나오고 있다.

예를 들어, Transfer Learning, Domain Adaptation, Semi-Supervised Learning, Weakly-Supervised Learning 등이 있고, 궁극적으로는 Unsupervised Learning으로 나아가기 위한 연구들이라고 볼 수도 있겠다. 그리고 **Self-supervised Learning**도 그 중의 하나로서 label이 없는 데이터를 사용한다. 

따라서, 주어진 domain (또는 데이터)에 알맞는 pretrined model이 없는 경우 **Self-supervised Learning**를 통해서 pretraining이 가능하며, label이 되어 있지 않은 데이터가 매우 많은 경우에 대해서도 **Self-supervised Learning**은 효율적으로 사용될 수 있다고 생각한다.


### 1. Pretext task 

이는 self-supervised learning을 위한 기법으로서, 사용자가 새로운 문제(pretext task)를 정의하고, 이에 대한 정답도 지정해준다. 그리고 모델을 학습하게 하여 주어진 데이터에 대해 이해를 시키고, 이를 downstream task로 transfer learning하는 과정을 수행하는 것이다.



### 2. Contrastive learning

이는 앞선 pretext task와는 또 다른 기법으로서, 같은 image에 서로 다른 augmentation을 한 positive pair의 feature representation은 거리가 가까워 지도록 학습을 하고, 다른 image에 서로 다른 augmentation을 한 negative pair의 feature representation은 거리가 멀어지도록 학습을 시키는 것이다. 

* [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf) -- MOCO

* [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf) -- SimCLR

* [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/pdf/2003.04297.pdf) -- MOCO v2

* [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf) -- SimCLR v2

* [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)





## References

* [awesome self-supervised learning](https://github.com/jason718/awesome-self-supervised-learning)