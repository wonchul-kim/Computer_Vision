## YOLO v3

* Fully convolutional netowrk
    * Convolutional layer, shortcut layer, upsample layer, route layer, yolo layer로 총 5개로 구성
        * yolov3.cfg 참조
        * shortcut layer: resnet의 skip connection과 동일
            * from: shortcut layer의 출력 이전 layer에서의 feature map과 shortcut layer의 3번째 뒤에 있는 layer를 추가
        * upsample layer: bilinear upsampling로 이전의 layer의 feature map을 upsample
        * route layer
            * 하나의 값을 가지는 경우, route layer로부터 변수만큼 떨어져 위치한 layer의 feature map을 출력
            * 두개의 값을 가지는 경우, route layer로부터 변수만큼 떨어져 위치한 layer들의 feature map을 연결하여 출력
        * yolo layer: 실제 학습을 진행하는 layer가 아닌 yolo 알고리즘의 detection에 대한 파라미터 정보를 내재
        <br/>

    * pooling을 사용하지 않고, stride=2인 convolutional layer를 사용하기 때문에 pooling으로 인한 low-level features의 정보손실을 방지
    * 입력 이미지의 크기에 영향을 받지 않지만, 일정한 크기의 입력 사이즈로 고정
        > 이는 header문제?
        > 또는 batch로 하여금 GPU로 병렬처리를 하는데에 있어서 용이하기 때문?

    * 1x1 convolutional layer를 사용함으로써 feature map (heatmap)으로 output을 생성
        * 각 cell(pixel 또는 unit/neuron)마다 anchor가 적용되어 object를 예측
        * output에서 channel의 크기는 B*(5 + C)
            * B: number of anchor boxes
            * C: number of class
            * 5: center coordinates($t_x, t_y, t_w, t_h$), objectness score
        > 각 cell마다 예측을 진행하는 거이기 때문에 어떻게 보면 cell크기 만큼의 window로 sliding window기법이라고 봐도 무방하지 않을까?

        * 일반적으로 416x416 이미지에 대해서 52x52, 26x26, 13x13의 feature map 도출
            * 최종적으로 ((52x52) + (26x26) + 13x13)) x 3 = 10647 개의 bounding box 예측
</br></br>

* Bounding box estimation
    * $b_x = \sigma(t_x) + c_x$
    * $b_y = \sigma(t_y) + c_y$
    * $b_w = p_we^{t_w}$
    * $b_h = p_he^{t_h}$
    </br>
    </br>
    * $b_x, b_y, b_w, b_h$: 예측한 bounding box 좌표 및 크기
        * $b_w, b_h$: 실제 이미지의 높이와 넓이로 normalize되어 있으므로 나중에 복원 필요
    * $t_x, t_y, t_w, t_h$: 신경망의 output
    * $c_x, c_y$: grid의 좌측 상단의 좌표
    * $p_w, p_h$: anchor box 크기
    * $\sigma()$: sigmoid func.
        * 이는 yolo가 bounding box의 중심좌표가 아닌 각 grid/cell의 좌측상단 좌표로부터의 offset을 예측하는 것이기 때문에 0 ~ 1의 범위에서 결과 도출
</br></br>

* Objectness score
    * bounding box에 object가 포함되어 있을 확률
    * `sigmoid function` 활용
</br></br>

* Class confidence
    * 검출된 객체가 특정 class에 속할 확률
    * `softmax`가 아닌 `sigmoid` 활용
        * `softmax function`은 calss가 상호 배타적으로 가정하기 때문
        * `woman`과 `person` 중 하나에 속해야함
</br></br>

* Non maximum supression
https://deep-learning-study.tistory.com/403
