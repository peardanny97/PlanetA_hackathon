# hackathon_ocean

해수면 높이와 표층 수온의 수평적 공간 분포로부터 특정 위치에서 수온 수직 분포를 예측하기

따라서 이 문제는 수온의 수직 분포와 관련된 해수면 높이, 그리고 표층 수온 사이의 관계를 학습시켜

해수면 높이와 표층 수온의 수평적 공간 분포로부터 수온의 수직 분포를 예측하는 모델 만들기

해수면 높이의 수평적 공간 분포와 표층 수온의 수평적 공간 분포가 predictor로 주어지며

그 공간 분포 내 특정 위치의 수온 수직 분포가 target

![image](https://user-images.githubusercontent.com/37990408/229995589-b73a6f86-18ff-4232-abd7-3db37968d5ee.png)

### Data 
•	1993-2021 기간 동안 월별 인공위성 관측 해수면 높이(predictor)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	1993-2021 기간 동안 월별 인공위성 관측 표층 수온(predictor)
![image](https://user-images.githubusercontent.com/37990408/229995754-abbeae8a-3e8c-4ad1-8ba0-6909a2c30a9d.png)
![image](https://user-images.githubusercontent.com/37990408/229995763-ea7ba31a-cf59-4e89-b316-7ba7d9c1c64a.png)

## Data Cleansing & Pre-Processing

sea surface temperature dataset(이하 sst dataset)과 sea level anomaly dataset(이하 sla dataset)에서 

latitude, longitude의 좌표가 서로 다른 것을 확인함 

이를 더 균일하게 좌표가 설정된 sst datset의 좌표에 맞춰 sla dataset을 coordinate하기로 결정

기존에 존재하지 않는 좌표에 대해 sla dataset을 만들어야 하기에 이를 위해 scipy의 interpolation을 사용

이 때에 육지를 나타내는 Nan 값은 제외하고 interpolation을 진행

interpolation된 sla dataset에 대해 sst dataset에서 육지인 곳을 참고하여 해당 data를 다시 Nan 값으로 처리하는 과정을 거침


target이 되는 수온 수직 분포 data의 경우 누락된 값이 많은 것을 확인, 해당 data를 년도별, 월별로 정렬

누락된 값의 경우 주어진 data에서 바다의 온도가 영하가 될 수 없다는 점을 참고하여 -1을 할당

target의 수직 분포에서 중간에 누락된 값이 많아 loss 계산이 잘 되지 않을 것을 고려하여 

해당 값을 시작 지점인 0m에서 -1인 값을 제외하고 양 끝 지점에서 관측치가 있는 수온 분포에 대해

그 사이의 누락된 값을 numpy의 interpolation을 진행함

## Exploratory Data Analysis

해양 표층 수온의 data의 경향성을 확인하기 위해 분기별 sst data를 plot

