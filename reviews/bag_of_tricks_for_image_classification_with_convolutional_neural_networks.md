

## Abs. & Intro.

본 논문은 그 동안 image classification의 성능이 많이 향상되었지만, 이는 단순히 neural network architecture의 발전으로만 가능했던 것은 아니다.
아래와 같이 training procedure refinements와 같은 source code나 구현방법에 의해서 나타나는 여러 기법에 의해서 가능했던 것이며, 이를 체계적으로 ablation study로 증명한다. (물론, 경험에(empirical) 의한 것도 포함된다.)

**training procedure refinements**

* data augmentation methods
* optimization methods
* loss functions
* data preprocessing

**model architecture refinements**

* stride size
* learning rate schedule 

따라서, 위의 사항에 대해 여러 종류의 networks를 empirical evaluation한 것이며, 결과적으로는 성능은 향상되었지만 연산량/연산시간은 늘어나지 않았다. 그리고 특정 network에 대해서는 generalization 성능이 향상되어 object segmetation과 semantic segmentation의 dataset에 대한 transfer learning의 성능도 향상됨을 보여준다.

## Training Procedure

#### Baseline training procedure

1. Randomly sample an image and decode it into 32-bit floating point raw pixel values in [0, 255].
2. Randomly crop a rectangular region whose aspect ratio is randomly sampled in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped region into a 224-by-224 square image.
3. Flip horizontally with 0.5 probability.
4. Scale hue, saturation, and brightness with coefficients uniformly drawn from [0.6, 1.4].
5. Add PCA noise with a coefficient sampled from a normal distribution N(0, 0.1).
6. Normalize RGB channels by subtracting [123.68, 116.779, 103.939] and dividing by [58.393, 57.12, 57.375], respectively.

* 모델의 모든 layer(Conv, FC layer)의 weight는 Xavier Initialization으로 진행하고, Batch Normalization의 $\gamma$는 1로, $\beta$는 0으로 Initialization한다. 

> In particular, we set the parameter to random values uniformly drawn from [-a, a], where $a = \square{6/(d_{in} + d_{out})}$. Here, $d_{in}, d_{out}$은 input과 output의 channel size. 음,,, xavier initialization에서 uniform distr.으로 파라미터를 초기화하는 과정이 있는 듯?????????????????????????????

* Optimizer는 NAG(Nesterov Accelerated Gradient)을 사용한다.

* 학습환경은 8개의 GPU, Batch Size=256, 120 Epoch, Initial Learning Rate = 0.1(매 30, 60, 90번째 epoch마다 1/10을 해주는 Step Decay)


#### Baseline validation procedure

1. Resize each image's shorter edge to 256 pixels while keeping its aspect ratio.
2. Crop region into a 224-by-224.
3. Normalize RGB channels by subtracting [123.68, 116.779, 103.939] and dividing by [58.393, 57.12, 57.375], respectively.


## Efficient Tranining

그간 GPU의 스펙이 많이 개선되어, lower numerical precision과 larger batch size로 학습시키는 것이 더 효율적이게 되었다. (model acc.의 감소없이) 즉, 속도와 정확도를 모두 개선시키는 것이 가능하다.

#### Large batch training

학습에 있어서 batch size가 크면, 학습 결과는 좋지 못할 수 있다. 그렇기 때문에 convex problem에 있어서는 convergence rate는 batch size에 반비례한다. 다시 말해서, 동일한 epoch 동안 batch size가 더 큰 경우, validation accuracy가 batch size가 작은 경우보다 더 작다고 한다. 이는 heuristics에 의한 많은 논문에서 제시한바가 있다고 한다. 

1. Linear scaling learning rate

mini-batch SGD에서 gradient descending은 각각의 batch에서 학습되는 data는 randomly sampling되기 때문에 사실상 random process라고 할 수 있다. 

> Increasing the batch size does not change the expectation of the stochastic gradient but reduces its variance. 에서 expectation of the stochastic gradient란???????

그리고 batch size를 늘리는 것은 stochastic gradient의 variance를 줄일 수 있고, 이는 gradient의 noise를 줄이는 것과 동일하므로, learning rate를 크게 할 수 있는 이점을 줄 수 있다. 

어떤 논문에서는 learning rate를 batch size의 크기에 정비례하는 실험을 통해 증명하였다.

> 학습 과정에서 learning rate를 변화하는 것은 일반적인데, batch size를 변화시킬 수 있는가? 변화시켜도 되는가??????????????????/

2. Learning rate warmup

초기 몇 epoch동안 매우 작은 learning rate(=~ 0)로 시작하여 initial learning rate로 linear하게 증가시키는 과정을 의미한다.

```
assume we will use the first $m$ batches (e.g. 5 data epochs) to warm up, and the
initial learning rate is $\eta$, then at batch $i$, $1 ≤ i ≤ m$, we will set the learning rate to be $i\eta/m$.
```


