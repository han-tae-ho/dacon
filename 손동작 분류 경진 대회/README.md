# π μλμ λΆλ₯ κ²½μ§ λν
- λν μκ° : μ΄ 4κ°μ class λ‘ μ΄λ£¨μ΄μ§ λ°μ΄ν°λ₯Ό λΆλ₯

- λν κ²°κ³Ό : Accuracy  : 0.87668 ( 44μ / 432ν )

- μ£Όμ κΈ°μ  : Lightgbm, catboost, CNN, Ensemble

- νΌλλ°±

  - Configs λ€ yamlλ argλ€λ‘ κ΅¬νν΄μ μ½λ ν΅ν©νκΈ°.

***
## β EDA

- train data κ°―μκ° test dataμ λΉν΄ νμ ν μ μ. -> μ¦κ° νμ

***
## β Model

- Lightgbm + Catboost + CNN
- weight = [0.3, 0.3, 0.4]

***

## β Train
```
python train.py
```

***
## β Eval
```
python eval.py
