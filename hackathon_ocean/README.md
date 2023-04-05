# hackathon_ocean

해수면 높이와 표층 수온의 수평적 공간 분포로부터 특정 위치에서 수온 수직 분포를 예측하는 모델 제작이 목표

따라서 이 문제는 수온의 수직 분포와 관련된 해수면 높이, 그리고 표층 수온 사이의 관계를 학습시켜 해수면 높이와 표층 수온의 수평적 공간 분포로부터 수온의 수직 분포를 예측하는 모델 만들기

해수면 높이의 수평적 공간 분포와 표층 수온의 수평적 공간 분포가 predictor로 주어지며 그 공간 분포 내 특정 위치의 수온 수직 분포가 target

![image](https://user-images.githubusercontent.com/37990408/229995589-b73a6f86-18ff-4232-abd7-3db37968d5ee.png)

### Data 
•	1993-2021 기간 동안 월별 인공위성 관측 해수면 높이(predictor)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	1993-2021 기간 동안 월별 인공위성 관측 표층 수온(predictor)
![image](https://user-images.githubusercontent.com/37990408/229995754-abbeae8a-3e8c-4ad1-8ba0-6909a2c30a9d.png)
![image](https://user-images.githubusercontent.com/37990408/229995763-ea7ba31a-cf59-4e89-b316-7ba7d9c1c64a.png)

## Data Cleansing & Pre-Processing

sea surface temperature dataset(이하 sst dataset)과 sea level anomaly dataset(이하 sla dataset)에서 latitude, longitude의 좌표가 서로 다른 것을 확인함 
이를 더 균일하게 좌표가 설정된 sst datset의 좌표에 맞춰 sla dataset을 coordinate하기로 결정

기존에 존재하지 않는 좌표에 대해 sla dataset을 만들어야 하기에 이를 위해 scipy의 interpolation을 사용, 이 때에 육지를 나타내는 Nan 값은 제외하고 interpolation을 진행함.
interpolation된 sla dataset에 대해 sst dataset에서 육지인 곳을 참고하여 해당 data를 다시 Nan 값으로 처리하는 과정을 거침


target이 되는 수온 수직 분포 data의 경우 누락된 값이 많은 것을 확인, 해당 data를 년도별, 월별로 정렬, 누락된 값의 경우 주어진 data에서 바다의 온도가 영하가 될 수 없다는 점을 참고하여 -1을 할당

target의 수직 분포에서 중간에 누락된 값이 많아 loss 계산이 잘 되지 않을 것을 고려하여 해당 값을 시작 지점인 0m에서 -1인 값을 제외하고 양 끝 지점에서 관측치가 있는 수온 분포에 대해 그 사이의 누락된 값을 numpy의 interpolation을 진행함

## Exploratory Data Analysis

해양 표층 수온의 data의 경향성을 확인하기 위해 분기별 sst data를 plot
![image](https://user-images.githubusercontent.com/37990408/229997553-3e773462-c3a8-48e5-bccb-ecc8c5ca2702.png)
<br><center>(1993년 2월, 5월, 8월, 11월의 표층 수온 데이터)</center>

해수면 높이 자료의 data의 경향성을 확인하기 위해 분기별 sla data를 plot
![image](https://user-images.githubusercontent.com/37990408/229997927-c6ab8ec6-fcaf-4824-97dd-fd17173bf175.png)
<br><center>(1993년 2월, 5월, 8월, 11월의 해수면 높이 데이터)</center>

해양 표층 수온, 해수면 높이 모두 위도, 경도, 날짜에 따라 크게 변한다는 경향성을 확인할 수 있음

target data와 train data사이의 관계를 파악하기 위해 비슷한 sst data에 대해 다른 sla data를 갖는 지점의 수온 수직 분포와 비슷한 sla data에 대해 다른 sst data를 갖는 수온 수직 분포를 엑셀을 통해 확인한 결과

sst의 경우 표층 수온이기 때문에 수온의 수직분포에서 시작점을 결정하며, sla의 경우 수온의 온도 변화 기울기에 영향을 미치는 것을 알 수 있음

## Feature Engineering & Initial Modeling

MLP와 CNN의 총 두가지 method를 시도해 보기로 결정

### MLP

MLP의 경우 feature는 표층 수온, 해수면 높이, 위도, 경도, 월을 feature로 사용하기로 결정, 각각을 flatten하여 vector로 만들어 주었으며 위도, 경도, 월의 경우 normalization을 거침. 각 feature vector를 concat하여 model에 feed함

Hidden dimension이 256인 2 layer MLP를 사용하였음, 마지막에 깊이 data인 14개의 vector가 출력되어야 하기에 Flatten한 후 linear layer를 거치도록 함

### CNN 

CNN의 경우 MLP와 마찬가지로 같은 feature를 사용, 각각을 2\*2 layer로 만들어 주었으며 위도, 경도, 월 layer의 경우 normalization을 거침. 해당 layer를 stack하여 2\*2\*5의 5개의 채널을 갖는 feature layer를 구성함

2D convolution을 사용했으며 Batch normalization과 Relu activation function을 사용, 이 때에 layer의 크기가 2\*2로 작기 때문에 더 작아지지 않도록 padding을 통해 kernel size를 2\*2로 고정시키고, dropout 등의 방법은 사용하지 않음. 마지막에 MLP와 마찬가지로 Flatten한 후 linear layer를 거치도록 함. 

## Model Tuning & Evaluation

Optimizer는 Adam을 이용하였고, scheduler는 CosineAnnealingScheduler를 이용

loss의 경우 Mean Squared Error loss function을 사용하였으며 label의 데이터 전처리 과정에서 interpolate된 길이가 작을수록 weight를 더 크게 주었음. 또한 처음 표층에 가까운 수온 분포에 대해 틀리게 될 경우 어긋나는 양상을 보였기에 처음의 포인트들에 더욱 가중치를 주는 방식을 택함.

## Conclusion & Discussion

학습 결과 MLP를 사용한 model의 경우 6.6123의 validation loss를 가지며 CNN 을 사용한 model의 경우 3.8907를 갖는 것을 확인
<br>학습한 model은 result 폴더 안에 pt 파일로 저장되어 있음

더 낮은 validation loss를 갖는 CNN model에 대해 true label과 prediction label에 대한 plotting을 진행한 결과는 아래와 같음
![image](https://user-images.githubusercontent.com/37990408/230000164-1abc41b0-9422-4192-9bdb-d74e2d515fc7.png)
<br>interpolate되지 않은 true label에 대한 prediction plot
![image](https://user-images.githubusercontent.com/37990408/230000246-26a7b7e8-b6b3-45f2-b405-210aa2e370ae.png)
<br>결측치가 6개인 true label에 대한 prediction plot

intepolate되지 않은 true label에 대해서는 어느 정도의 경향성을 잘 따라가고 있음을 알 수 있음.

결측치가 존재하는 true label에 대해서는 결측치가 존재하기 전까지의 label에 대해 robust하게 model을 예측하고 있는 모습을 확인할 수 있음.

해당 model을 fitting하는 데에 있어 결측치가 가장 큰 문제점을 야기함. 더 accurate한 target data가 존재한다면 CNN이 더 잘 fitting될 수 있을 것이라 예상 가능함.

