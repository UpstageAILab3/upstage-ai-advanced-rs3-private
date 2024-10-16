import pandas as pd
import numpy as np
from xgboost import XGBRanker
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# 데이터 전처리 함수
def preprocess_data(user_item_weights):
    # XGBRanker에서 사용할 feature와 target 분리
    features = ['price', 'date_weight', 'frequency', 'monetary', 'cluster', 'brand_encoded']
    target = 'event_weight'

    X = user_item_weights[features]
    y = user_item_weights[target]

    return X, y

# XGBRanker 모델 훈련 함수
def train_xgbranker(user_item_weights):
    X, y = preprocess_data(user_item_weights)

    # 사용자별로 아이템 수 그룹화
    group_sizes = user_item_weights.groupby('user_id').size().tolist()

    # XGBRanker 모델 정의 및 훈련
    ranker = XGBRanker(
        objective='rank:pairwise',  # pairwise rank objective
        learning_rate=0.0778,
        alpha=0.0670,
        base_score=0.5627,
        gamma=4.6261,
        max_depth=6,
        n_estimators=350,
        tree_method='hist',  # GPU 가속 사용
        device='cuda',
        # predictor='gpu_predictor',  # GPU 사용
        # n_gpus=-1,  # 시스템의 모든 GPU 사용
        n_jobs=32 # CPU 병렬 처리 (32코어 사용)
    )

    # group 매개변수에 각 user_id가 가진 item의 개수 전달
    ranker.fit(X, y, group=group_sizes)

    return ranker


# 추천 생성 함수
def recommend_items_xgboost(user_item_weights, ranker, top_n=10, batch_size=1000, num_threads=32):
    # 사용자 목록
    user_ids = user_item_weights['user_id'].unique()

    recommendations = {}

    # 사용자별로 추천 생성 작업을 수행하는 함수
    def recommend_for_user(user_id):
        user_data = user_item_weights[user_item_weights['user_id'] == user_id]
        if not user_data.empty:
            X_user = user_data[['price', 'date_weight', 'frequency', 'monetary', 'cluster', 'brand_encoded']]

            # XGBRanker로 예측한 가중치
            user_data.loc[:, 'predicted_weight'] = ranker.predict(X_user)

            # 상위 top_n개의 item_id 추천
            top_items = user_data.sort_values(by='predicted_weight', ascending=False)['item_id'].head(top_n).tolist()
            return user_id, top_items
        return user_id, []

    # 사용자 데이터를 배치 단위로 분할하여 처리
    user_id_batches = np.array_split(user_ids, len(user_ids) // batch_size + 1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # tqdm을 사용하여 각 배치에 대해 병렬 처리 진행 상황 표시
        for user_id_batch in tqdm(user_id_batches, desc="Processing batches", leave=True):
            futures = {executor.submit(recommend_for_user, user_id): user_id for user_id in user_id_batch}
            
            # 추천 생성 진행 상황을 표시하며 추천 생성
            for future in tqdm(as_completed(futures), total=len(user_id_batch), desc="Generating Recommendations", leave=True):
                user_id, top_items = future.result()
                recommendations[user_id] = top_items

    return recommendations

if __name__ == "__main__":
    # 데이터 불러오기
    user_item_weights = pd.read_csv('../data/reduce_total_train.csv')

    # XGBRanker 모델 훈련
    print("Training XGBRanker Model...")
    ranker = train_xgbranker(user_item_weights)

    # 모델 저장
    with open('xgbranker_model.pkl', 'wb') as f:
        pickle.dump(ranker, f)

    # 사용자별 추천 생성 (상위 10개)
    print("Generating Recommendations...")
    recommendations = recommend_items_xgboost(user_item_weights, ranker, top_n=10)

    # 결과 저장
    result_df = pd.DataFrame(recommendations.items(), columns=['user_id', 'item_id_list'])
    result_df.to_csv('xgbranker_recommendations.csv', index=False)
    print("추천 결과가 저장되었습니다!")
