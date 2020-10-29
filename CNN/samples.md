### Samples for Computer Vision


#### Postivie & Negative sample

object detection에서 실제로 detection하고자 하는 object를 호함하는 데이터를 positive sample이라고 하며, 이외의 object에 대해서 negative sample이라고 한다. 예를 들어, face detection 알고리즘에서는 얼굴을 postiive sample이라고 하며, 그 외의는 background로서  negative sample이라고 한다.

이 때, **hard negative sample**은 실제로는 negative인데 positive로 예측하기 쉬운 데이터를 의미하며, 이에 대한 **hard negative mining**은 hard negative 데이터를 학습데이터로 사용하기 위해 모으는 작업을 의미한다. 그리고 이로 인해 얻은 데이터를 원래의 데이터에 추가해서 재학습하면 false negative에 대한 오류를 줄일 수 있다.


일반적으로 object detection에서 false negative 가 발생하는 이유는 positive sample에 비해 negative sample이 굉장히 많은 클래스 불균형 문제 때문이다. 이러한 클래스 불균형 문제는 성능에 끼치는 영향이 크다. 예를 들어서, 얼굴을 인식하는 프로젝트에 대해서 얼굴에 대한 bounding box(ROI)를 하나의 positive sample이라고 할 때, 그 이외의 background인 negative sample이 더 많은게 일반적이다.  


* positive samples:
detection하고자 하는 object를 포함하는 이미지의 ROI

* negative samples:
그 이외의 backgroud에 대한 이미지의 ROI

> hard negative samples:
실제로 negative sample인데 positive로 예측하기 쉬운 sample

> easy negative samples:
실제로 negative samples이고, negative로 예측하기 쉬운 sample
