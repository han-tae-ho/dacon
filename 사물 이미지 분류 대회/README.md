# ✈ 사물 이미지 분류 대회
- 대회 소개 : 총 10개의 class 로 이루어진 데이터를 분류

- 대회 결과 : Accuracy  : 0.8712 ( 26위 / 460팀 )

- 주요 기술 : Computer Vision, Image Classification, SRCNN + Efficientnetb0(or Resnet34), Pesudo Labeling, TTA

- 피드백

  - Meta Pesudo Labeling 구현 및 적용을 안한 점 (그냥 Pesudo Labeling 사용)

  - Teacher 모델과 student 모델 dropout layer 수치 다르게 돌려봤으면 ?? (labeled data의 dropout이 더 높게 - 논문참고) 

  - 데이터 EDA 결과, 이미지 resolution이 매우 낮은 것을 확인 -> HRGAN 사용 후 classification model 적용 어떨까 생각.  

    but, 대회 규정 상 pretrained model 사용 금지, 외부 데이터 사용 금지 이유로 불가능해서 아쉽다.

  - row resolution개선하기 위해 classifier model 앞에 SRCNN layer 추가. -> 추가을 했을 때와 안했을 때 비교 못한 것(성능 향상 되었을까 ?)

  - DataAug나 Hyper parameter 수정 시 optuna 이용 해보기, pesudo label 사용시 randomAug 적용해보기

  - Configs 들 yaml나 arg들로 구현해서 코드 통합하기.

***
## ✔ EDA

- 각 class 별 5000장 씩 -> balanced data

- very low resolution -> SR network 사용 -> but 대회 규정 상 불가 -> layer 만 사용


***
## ✔ Model

- SRCNN + Efficientnetb0

- loss : CrossEntropyLoss

- optim : Adam

- Scheduler : ReduceLROnPlateau

- LEARNING_RATE = 0.002

- EPOCH = 40

- BATCH_SIZE = 64

***
## ✔ Requirements
```
pip install -r requirements.txt
```

***
## ✔ Train
```
python train.py
```

***
## ✔ Eval
```
python eval.py
