# 🖐 손동작 분류 경진 대회
- 대회 소개 : 총 4개의 class 로 이루어진 데이터를 분류

- 대회 결과 : Accuracy  : 0.87668 ( 44위 / 432팀 )

- 주요 기술 : Lightgbm, catboost, CNN, Ensemble

- 피드백

  - Configs 들 yaml나 arg들로 구현해서 코드 통합하기.

***
## ✔ EDA

- train data 갯수가 test data에 비해 현저히 적음. -> 증강 필요

***
## ✔ Model

- Lightgbm + Catboost + CNN
- weight = [0.3, 0.3, 0.4]

***

## ✔ Train
```
python train.py
```

***
## ✔ Eval
```
python eval.py
