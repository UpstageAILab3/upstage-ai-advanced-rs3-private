# Commerce Purchase Behavior Prediction | 커머스 상품 구매 예측 | 경진대회
## 팀원
|![김나리](https://avatars.githubusercontent.com/u/137861675?v=4) |
:--------------------------------------------------------------: 
|[김나리](https://github.com/narykkim)|
|EDA, Pre-processing, Modeling|

### EDA <다른 조원들과 다른점>
- category_code의 비율이 너무 차이가 나기 때문에 이걸 지우고 brand를 살림.

### Boost 모델을 위한 Feature engineering
- 핵심적인 생각
    - 한 유저가 한 아이템을 어떻게 생각하는지 수치화 해보자. 
    - event_type을 아이템별로 묶을 수 있도록 각각 값에 가중치를 부여한 후 합하여, 이 값을 타겟으로 삼아 공략.
- event_weight :
    - view, cart, purchase를 전체의 갯수 대비 비율의 역수로 하여 각각의 weight를 다르게 부여함.
    - 한 유저가 view만 많이 했다면, 가중치를 적게 받고 한번이라고 구매를 했다면 높은 가중치를 받게된다.
    - 구매를 하지 않고 view나 cart만 있는 사용자들은 event_weight가 높은 3개의 아이템만을 두고 나머지는 삭제.
    - 결과적으로 좋은 점 중 하나는 데이터의 수가 줄어든다는 것임. (약 800만개 -> 약 160만개)
- date_weight :
    - event_type을 수치화하면서 시간과 아이템과의 관계를 정립할 필요가 생김. 
    - event_time으로 정렬 후, 2월에 상호작용이 일어난 것에 대해 weight를 주는 열인 date_weight를 설정함.
    - date_weight를 설정할 때, view, cart는 5점을, purchase는 50달러 이상이면 2점, 50달러 미만이면 5점을 부여. - 재구매에 대한 가능성.
- 위가 item에 관한 정보라면 아래는 user에 관한 정보.
- monetary : 각각의 user_id가 구매한  총 금액.
- frequency : 각각의 user_id가 상호작용한 횟수 (type과 item에 상관없이 셈.)
- cluseter : monetary, frequency를 이용하여 k-mean clustering 하여 0,1,2 이렇게 3그룹으로 나누고, 이상치들은 묶어서 3을 부여함.
- brand


![](https://raw.githubusercontent.com/UpstageAILab3/upstage-ai-advanced-rs3-private/refs/heads/main/narykim/clustering.png?token=GHSAT0AAAAAACX7W7IRGOHJRURKEO7B5L5WZY2K57Q)
![](https://raw.githubusercontent.com/UpstageAILab3/upstage-ai-advanced-rs3-private/refs/heads/main/narykim/feature_importance.png?token=GHSAT0AAAAAACX7W7IQF7OLZZ3YFWFKJ3QYZY2K6RQ)


### 사용 모델
- XGBoostRanker
- CatboostRanker
- ALS
- GRU4Rec
- SASRec

### 특이사항?
- XGBoostRanker와 CatboostRanker
  - event_weight값을 타겟으로 할 경우 한 아이디당 10개미만의 아이템들과 상호작용하는 경우가 많이 있어서, 이럴 경우 결과값이 아이디당 딱 10개로 떨어지지 않음.
  - 10개를 채우기 위해서는 앙상블이 불가피함. - 초반에는 베이스코드에서 제공해준 als를 이용, 대회가 진행되면서 리더보드의 점수가 좋은 output으로 대체.
  - 연산시간이 16시간정도로 오래걸려, 하이퍼파라미터 튜닝시에는 valid data를 2월 27, 28, 29일의 구매자 아이디로만 구성하여 빠르게 테스트. (삭제할까말까 고민중 ㅎㅎ)
- recbole을 이용하여 GRU4Rec과 SASRec을 손쉽게 다룰수 있었음.
- GRU4Rec
  - 세션 기반 추천 시스템에 자주 사용되는 모델임. GRU(게이트 순환 유닛)를 사용해 순차적인 사용자 행동을 학습하고, 세션 내의 사용자 아이템 상호작용을 기반으로 다음에 추천할 아이템을 예측함. 즉, 세션 안에서 시간 순서대로 발생하는 행동들을 고려해 다음에 사용자가 클릭할 아이템을 예측하는 방식임.
  - 그래서 위의 피쳐들을 사용하지 않고, user_id, item_id, event_time, event_session을 이용하여 연산함.
- SASRec은 베이스코드와 다르게 Feature engineering에서 소개한 feature들을 사용하여 연산해봄.

### 결과 (실험한 시간순으로 정리했고 당연히 빼도 됩니다. 학습블로그 정리용으로 써놓았습니다. 제가 한 실험과 제 실험에 도움을 준 output들의 점수를 중심으로 적은 점수라서 엄청 제 중심적입니다.)

- als (base code) : 0.0846 (0.0853)
- XGBRanker + als : 0.1211 (0.1210)
- XGBRanker(params tune) + als : 0.1221 (0.1214)
- CatRanker(params tune) + als : 0.1208 (0.1200)
  
- als(params tune) : 0.1059 (0.1058)
- CatRanker(params tune) + als(params tune) : 0.1219 (0.1205)
- XGBRanker(params tune) + als(params tune) : 0.1232 (0.1219)
  
- GRU4Rec + populer top10 : 0.0980 (0.0980)
- GRU4Rec + als(params tune) : 0.0929 (0.0929)
- SASRec + populer top10 : 0.0870 (0.0876)

- LMF(params v1) :0.0944 (0.0925)
- LMF(params v2) :0.1141 (0.1132)
- LMF(params v3) : 0.1214 (0.1213)
  
- CatRanker + LMF : 0.1313 (0.1304)
- XGBRanker + LMF : 0.1325 (0.1318) BEST!
- XGBRanker_als + CatRanker_als + LMF using ranking : 0.1324 (0.1317)

### 시도한 점
- 처음부터 우리조의 목표가 최대한 많은 모델을 돌려보자였기 때문에 많이 돌려보려고 노력하였고, 모두 각자의 모델을 잘 돌려서 좋은 결과가 나왔던것 같다.
  
### 아쉬운 점
- Boost 계열이 시간이 오래걸려서 개인적으로는 CF, MF 모델들에 대한 여러 실험을 못해봤는 데, 해보고 싶다.
