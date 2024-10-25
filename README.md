[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)
# Commerce Purchase Behavior Prediction | 커머스 상품 구매 예측 | 경진대회
## Team


| ![박범철](https://avatars.githubusercontent.com/u/117797850?v=4) |![김나리](https://avatars.githubusercontent.com/u/137861675?v=4) |   ![조용중](https://avatars.githubusercontent.com/u/5877567?v=4) | ![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) ||
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|                       [박범철](https://github.com/Bomtori)             | [김나리](https://github.com/narykkim)             |                      [조용중](https://github.com/paanmego)             |            [최윤설](https://github.com/developzest)             |
|                            팀장, 발표, EDA, Pre-processing, Modeling                            |                            EDA, Pre-processing, Modeling                          |                 EDA, Pre-processing, Modeling                  |                            EDA, Pre-processing, Modeling                          | 

## 1. Competiton Info

### Overview

'Commerce Behavior Purchase Prediction' 대회는 사용자의 쇼핑 패턴을 분석하여 향후 1주일 동안 구매할 상품을 추천하는 것을 목표로 한다. 추천 시스템은 개인의 쇼핑 습관과 과거 구매 이력을 분석해 맞춤형 상품을 제안함으로써, 사용자의 경험을 개선하고 기업의 매출을 증가시킨다. 이커머스 추천 시스템 구축 과정은 데이터 전처리부터 모델 선택, PyTorch와 라이브러리 활용, Feature Engineering 및 예측 수행을 포함한다. 대회에서는 평가 지표에 최적화된 파이프라인을 개발하는 것이 중요하다. 또한, 현업에서는 어려울 수 있는 복잡한 구조나 다중 모델 앙상블도 높은 점수를 위해 고려할 수 있다.

### Timeline

- 2024년 10월 02일 : 대회시작
- 2024년 10월 08일 : 개별적으로 강의수강, EDA 진행
- 2024년 10월 10일 ~ 10월 23일 : 여러 모델 선정 및 파인 튜닝
- 2024년 10월 24일, 25일 : 결과 앙상블
- 2024년 10월 25일 : 대회 종료

## 2. Data descrption

### Dataset overview

- 학습데이터 : train.parquet

    - 19년 11월 1일부터 20년 2월 29일까지 4개월간의 데이터
    - 8,350,311개의 행으로 이루어져 있다.
    - user_id : 유저 id
    - item_id : 아이템 id
    - user_session : 사용자의 세션 ID. 사용자가 오랜 일시 중지 후 온라인 스토어로 돌아올 때마다 변경된다.
    - event_time : 이벤트가 일어난 시각(UTC기준)
    - category_code : 아이템의 카테고리 분류입니다.
    - brand : 아이템의 brand
    - price : 아이템의 가격
    - event_type : 이벤트의 종류
- 평가데이터
    - 20년 3월 1일부터 20년 3월 7일까지 일주일 간의 데이터.
    - 해당 기간 동안 유저가 구입한(event_type = 'purchase') 아이템 이력에 대한 데이터로 user_id와 item_id로 구성된다.
    - 평가데이터는 무작위 (50:50 random split)로 public, private dataset으로 나뉨.
    - public dataset
    - 대회 기간중 리더보드 점수 계산에 활용되는 정답 데이터.

- private dataset
    - 대회 종료후 최종 점수 계산에 활용되는 정답 데이터.
    - 평가 데이터에는 학습데이터에 포함된 유저와 아이템으로만 이뤄져 있다.
      
### Data Processing

- 중복 데이터 18개 존재하여 제거.

## 3. Modeling

- 협업필터링 기반 모델
- Boost 모델
- 시퀀스 기반 모델
- 하이브리드 모델

## 4. Result

### Leader Board

- 1등 : 0.1325 (0.1318)

### Presentation

- _Insert your presentaion file(pdf) link_
