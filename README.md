# IMBK_Bank_Customer_Churn_ML

- 프로젝트명: 고객 이탈 분류 ML 및 인사이트분석
| 기간 : 2026년 4월 10일 |

## Tech Stack

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

- 데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

## Data Preprocessing (데이터 전처리)

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

## EDA 및 해석
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

8. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

Model Comparison (Top 10)

┌──────────────────────────────┬──────────┬────────┬────────┬────────┐
│ Model                        │ Accuracy │  AUC   │ Recall │  F1    │
├──────────────────────────────┼──────────┼────────┼────────┼────────┤
│ LightGBM                     │ 0.8605   │ 0.8591 │ 0.4847 │ 0.5858 │
│ CatBoost                     │ 0.8600   │ 0.8644 │ 0.4730 │ 0.5790 │
│ Random Forest                │ 0.8622   │ 0.8533 │ 0.4552 │ 0.5736 │
│ Gradient Boosting            │ 0.8614   │ 0.8611 │ 0.4534 │ 0.5714 │
│ XGBoost                      │ 0.8526   │ 0.8447 │ 0.4761 │ 0.5681 │
│ AdaBoost                     │ 0.8508   │ 0.8427 │ 0.4620 │ 0.5579 │
│ Extra Trees                  │ 0.8532   │ 0.8461 │ 0.4399 │ 0.5492 │
│ Decision Tree                │ 0.7862   │ 0.6766 │ 0.4914 │ 0.4836 │
│ K-Nearest Neighbors          │ 0.8310   │ 0.7755 │ 0.3798 │ 0.4779 │
│ Quadratic Discriminant (QDA) │ 0.8370   │ 0.8083 │ 0.2975 │ 0.4264 │
└──────────────────────────────┴──────────┴────────┴────────┴────────┘

Top Models (Selected by PyCaret)
┌──────┬──────────────────────────────┐
│ Rank │ Model                        │
├──────┼──────────────────────────────┤
│ 1    │ LightGBM                     │
│ 2    │ CatBoost                     │
│ 3    │ Random Forest                │
│ 4    │ Gradient Boosting            │
└──────┴──────────────────────────────┘
- PyCaret 기반 모델 비교 결과 상위 4개 모델 선정

10. 인사이트제안
11. Reference
