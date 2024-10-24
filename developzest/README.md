# Commerce Purchase Behavior Prediction | 커머스 상품 구매 예측 | 경진대회
## 팀원
|![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) |
:--------------------------------------------------------------: 
|[최윤설](https://github.com/developzest)|
|EDA, Pre-processing, Modeling|

### 전처리
- 중복 데이터 18개 존재하여 제거.
  - 멘토링 시, test set에도 중복 데이터 존재 가능성이 있으므로 제거하지 않는 것을 추천한다고 했으나 ALS, SASREC 모두 중복 데이터를 제거한 결과의 리더보드 성능이 더 좋았기 때문에 제거함.

### EDA
- shoes가 압도적으로 비중이 높음.
- 신발 브랜드에 가전제품 및 칩셋 제조사가 섞여있어 brand 컬럼은 고려하지 않는 것으로 함.
- 특정 날짜에 사용자 상호작용 이상치 존재.
- 오후 시간대, 금 / 토 / 일요일에 주문이 많음.
- 전체 이벤트 중 대부분 view 이며 cart와 purchase 비중이 현저이 적음.
- 전체 상품의 카테고리 별 가격과 구매한 상품의 카테고리별 가격 분포 비교 시, 비교적 저렴한 제품 구매.

### 사용 모델
- 메모리 기반 CF, ALS, SASRec, BPR, LMF 모델링 함.
- 다양한 모델과 많은 실험을 하고자 했으며, Kaggle이나 Dacon의 추천시스템 대회에서 높은 순위를 차지한 모델에 대한 인사이트 참고.
- ALS가 baseline 코드여서 implicit 라이브러리에서 제공하는 다른 GPU, CPU 모델에 대해 호기심이 생겨 공부할 겸 해당 모델들로 실험하기로 함.

### 시도한 방법
- event_type이 'view'인 데이터를 제거하지 않음. 
- 사용자-아이템 간 상호작용을 나타내는 이진 레이블 열 생성 시, event_type 별로 다른 가중치를 적용.
- label 외 별도 feature engineering을 하지 않은 LMF의 Public ndcg@10: 0.1205 라 XGBoost / Catboost와 앙상블
- WandB의 sweep과 Optuna 활용하여 하이퍼 파라미터 튜닝 수행. 

### 아쉬운 점
- Recbole 라이브러리에서 제공하는 다양한 모델들에 대한 실험?ㅎ