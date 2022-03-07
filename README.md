# 🏆🏆 Dacon Competition 🏆🏆


***
## 🦽 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

- 대회 소개 : 병해 피해를 입은 작물 사진과 작물의 생장 환경 데이터를 이용해 작물의 병해를 진단하는 AI 모델을 개발
- 대회 결과 : F1 score : 0.779 ( 290위 / 1513팀 )
- 주요 기술 : Image Classification, Efficientnetb0
- 피드백
  - 대회 하루 전 참여하여 single model 만 돌려본 점이 아쉽다.
  - 시간이 없다보니 train, valid split 없이 train만 돌려서 overfitting이 난 것 같다.   
    (토론 게시판 보니 image만 이용한 singlemodel 0.85 정도 나옴)
  - 생활 환경 데이터를 활용 하지 않았다. 시간이 충분했다면 ? Catboost, XGboost, LightGBM 사용하지 않았을까??

***
## ✈ 사물 이미지 분류 대회
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

***
