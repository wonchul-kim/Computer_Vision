### 주요 내용 요약
* [핵심] 바운딩 박스를 인풋으로 주지 않아도, 오브젝트 디텍션 용도로 학습된 모델을 조금만 튜닝해서 오브젝트 위치 추적이 가능하다.
* 탐지 대상인 사물의 위치 정보가 없어도 CNN은 오브젝트 디텍터로서 기능한다.
* 그런데 분류를 위해 fully connected layer를 사용함으로써 이 기능이 사라진다.
* Network In Network나 GoogLeNet에서는 파라미터 수 최소화를 위해 fully connected 대신 Global Averag Pooling(이하 GAP)을 썼다.
* 그런데 이 GAP는 파라미터 수를 줄여 오버피팅을 방지하는 기능 외에도, 오브젝트의 위치 정보를 보존하는데 사용할 수 있다.
* GAP를 통해 특정 클래스에 반응하는 영역을 맵핑하는 Class Activation Mapping(이하 CAM)을 제안한다.
* 당시 제안된 다른 위치 추적 방식들에 비해 CAM은 한번에 (single forward pass) end-to-end로 학습할 수 있다.
* FC를 GAP로 대체해도 성능 저하가 크게 일어나지 않았으며 이를 보정하기 위한 아키텍쳐도 제안한다.
* Global Max Pooling은 탐지 사물을 포인트로 짚는 반면, GAP는 사물의 위치를 범위로 잡아내는 장점이 있다.
* 다른 데이터셋을 활용한 분류, 위치 특정, 컨셉 추출에도 쉽게 대입해서 사용할 수 있다.

