---
slug: titanic-logistic-regression
title: "타이타닉 생존자 예측 모델 구현하기"
authors: [silee]
tags: [데이터분석, 머신러닝, 튜토리얼]
---

<div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px'}}>
  <span>이 프로젝트는 Kaggle에서 제공하는 타이타닉 데이터를 이용해 생존 여부를 예측하는 과정을 보여줍니다. 주피터 노트북을 통해 데이터 탐색을 진행하고, <code>scikit-learn</code>의 로지스틱 회귀 모델을 사용해 예측 모델을 학습합니다.</span>
</div>

<!-- truncate -->

## 데이터 준비

- `titanic/` 폴더에 `train.csv`, `test.csv`, `gender_submission.csv` 파일이 있습니다.
- `01_titanic_EDA.ipynb` 노트북에서 Pandas로 데이터를 로드하고 기본 통계 정보를 확인합니다.

```python
import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("gender_submission.csv")
```

## 간단한 탐색

노트북에서 `train.info()`와 `train.describe()` 등을 사용해 각 열의 결측치와 분포를 확인합니다. 시본(Seaborn)과 맷플롯립(Matplotlib)으로 시각화해 승객 나이 분포 등을 파악합니다.

## 로지스틱 회귀 모델 학습

선택한 특징(`PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`)을 사용하여 로지스틱 회귀 모델을 학습합니다.

```python
from sklearn.linear_model import LogisticRegression
sel = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch']
X_train = train[sel]
y_train = train['Survived']
X_test = test[sel]

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## 예측 결과 저장

예측 결과는 `firstSub.csv`로 저장합니다.

```python
sub['Survived'] = pred
sub.to_csv("firstSub.csv", index=False)
```

이 파일은 Kaggle 제출 형식에 맞춰 생성된 것으로, 프로젝트 루트에 이미 포함되어 있습니다. 이를 이용해 Kaggle 대회에 바로 제출할 수 있습니다.

