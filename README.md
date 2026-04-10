## 고객 이탈 분류 ML 및 인사이트분석

## 2. 기간 : 
2026년 4월 10일 

## 3. Tech Stack

### Data Analysis
- pandas, numpy

### Visualization
- matplotlib

### Machine Learning
- scikit-learn

### Boosting Models
- XGBoost, LightGBM, CatBoost

### AutoML & Optimization
- PyCaret, Optuna

### Model Interpretation
- SHAP
---
## 4. 데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)
---
## 5. Data Preprocessing (데이터 전처리)

### 1. 불필요한 컬럼 제거
- 고객 고유 식별자(customer_id)는 예측에 불필요한 개별 식별 정보 제거

### 2. 결측치 처리
- 전체 데이터에서 결측치 여부를 확인 후 이상 없음 확인

### 3. 범주형 변수 인코딩
- country, gender 변수는 문자열 형태이므로 Label Encoding을 통해 수치형으로 변환

### 4. 데이터 분할
- train / validation 데이터를 8:2 비율로 분할
- stratify를 적용하여 클래스 불균형 유지

### 5. 스케일링
- StandardScaler를 사용하여 데이터 정규화 수행

--- 

## 6. EDA 및 해석
### 1. Active Member에 따른 이탈률 분석
<img width="500" height="400" alt="활성화 고객" src="https://github.com/user-attachments/assets/1fe8b4d5-8488-4981-b92d-2c6e7c48e7e4" />

- active_member 기준으로 고객을 그룹화하여 churn 비율을 비교
- 비활성 고객(0)의 이탈률이 활성 고객(1) 대비 약 2배 높게 나타남

### 2. 주요 인사이트 
- 고객 활동 여부(active_member)는 churn 예측에 핵심 변수
- 비활성 고객일수록 서비스 이탈 가능성이 높음

### 3. 비즈니스 활용 방안
- 비활성 고객을 조기에 식별하여 이탈 방지 전략 필요
- 고객 참여를 유도하는 기능 강화
  - 맞춤형 알림
  - 리워드 제공
  - 개인화 서비스 추천
---
## 7-1. AutoML (PyCaret)
- PyCaret을 활용하여 다양한 분류 모델을 자동으로 비교하고 성능을 평가함
- 클래스 불균형을 고려하여 Accuracy가 아닌 F1-score를 기준으로 모델을 선정

Model Comparison (Top 10)
  
| Rank | Model                        | Accuracy | AUC    | Recall | F1     |
|------|------------------------------|----------|--------|--------|--------|
| 1    | LightGBM                     | 0.8605   | 0.8591 | 0.4847 | 0.5858 |
| 2    | CatBoost                     | 0.8600   | 0.8644 | 0.4730 | 0.5790 |
| 3    | Random Forest                | 0.8622   | 0.8533 | 0.4552 | 0.5736 |
| 4    | Gradient Boosting            | 0.8614   | 0.8611 | 0.4534 | 0.5714 |
| 5    | XGBoost                      | 0.8526   | 0.8447 | 0.4761 | 0.5681 |
| 6    | AdaBoost                     | 0.8508   | 0.8427 | 0.4620 | 0.5579 |
| 7    | Extra Trees                  | 0.8532   | 0.8461 | 0.4399 | 0.5492 |
| 8    | Decision Tree                | 0.7862   | 0.6766 | 0.4914 | 0.4836 |
| 9    | K-Nearest Neighbors          | 0.8310   | 0.7755 | 0.3798 | 0.4779 |
| 10   | Quadratic Discriminant (QDA) | 0.8370   | 0.8083 | 0.2975 | 0.4264 |

Top Models (Selected by PyCaret)
| Rank | Model             |
|------|------------------|
| 1    | LightGBM         |
| 2    | CatBoost         |
| 3    | Random Forest    |
| 4    | Gradient Boosting|
- PyCaret 기반 모델 비교 결과 상위 4개 모델 선정

---

## 7-2. Hyperparameter Tuning (Optuna)
- 상위 모델을 대상으로 하이퍼파라미터 최적화 수행

<img width="400" height="200" alt="최적화" src="https://github.com/user-attachments/assets/a0028bd3-7e50-4409-b8dc-f840f75efb0f" />

---
## 7-3. Stacking
- 성능이 우수한 모델(LightGBM, CatBoost, XGBoost)을 Base Model로 구성
- Logistic Regression을 Meta Model로 활용하여 Stacking 수행
  
<img width="1000" height="550" alt="image" src="https://github.com/user-attachments/assets/2973173b-ac48-4219-95dd-7cc3245b54c4" />


---
## 7-4. SHAP value

<img width="1000" height="700" alt="분석결과" src="https://github.com/user-attachments/assets/f00382ed-b06f-439f-9ebf-d7b946183d39" />

## 7-5. SHAP Analysis

- 주요 영향 변수: age, active_member, balance, product_number, credit_score

- age: 값이 높을수록 이탈(churn) 확률이 증가하는 경향
- active_member: 비활성 고객일수록 이탈 확률이 높고, 활성 고객은 이탈하지 않는 방향으로 작용
- balance: 잔액이 높을수록 이탈 가능성이 증가하는 패턴
- credit_score: 낮을수록 이탈 확률이 높아지는 경향
- product_number: 값에 따라 이탈/비이탈 방향이 혼재되어 단일 변수로는 해석이 어려움

- credit_card, tenure, estimated_salary 변수는 SHAP 값이 0 근처에 분포하여 상대적으로 영향력이 낮음
---
## 8. 인사이트

- SHAP 분석 결과, product_number, age, active_member, balance가 주요 영향 변수로 확인됨

- product_number는 SHAP 값이 양·음 방향에 모두 분포하여 영향 방향이 일정하지 않으므로, 단일 변수 기준으로 고객을 판단하기보다 복합적인 변수 조합 기반 분석이 필요

- age, active_member, balance는 값에 따라 churn 예측 방향이 달라지는 패턴을 보이며, 고객 상태 및 행동 특성이 이탈에 중요한 영향을 미침

- 이를 기반으로 주요 변수 중심의 고객 세분화를 통해 이탈 가능성이 높은 집단을 식별하고, 해당 고객군에 대한 맞춤형 관리 전략 수립 필요

- 단일 기준이 아닌 다변수 기반의 세분화 전략을 적용할 경우, 고객 유지 전략의 효과를 더욱 높일 수 있음
---
## 9. Reference

- PyCaret: AutoML 기반 모델 비교 및 선택
- Optuna: 하이퍼파라미터 최적화
- SHAP: 모델 예측 해석 및 변수 영향 분석

- LightGBM, XGBoost, CatBoost: Boosting 기반 머신러닝 모델
- Scikit-learn: 기본 머신러닝 모델 및 Stacking 구현
- Pandas, NumPy: 데이터 처리 및 분석
- Matplotlib: 데이터 시각화

- 데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)
