import pandas as pd
import os
import json
import numpy as np
import random
import torch
from torch.nn.utils import clip_grad_norm_
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_gpu_usage
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calculate_valid_score(valid_result, valid_metric):
    """
    Calculate validation score based on validation result and metric.
    
    Args:
        valid_result (dict): A dictionary of validation results containing metrics
        valid_metric (str): Metric name to be used for validation score calculation
        
    Returns:
        float: Calculated validation score
    """
    valid_metric = valid_metric.lower()
    if valid_metric in valid_result:
        return valid_result[valid_metric]
    else:
        raise ValueError(f"Valid metric {valid_metric} not found in {valid_result.keys()}")

def preprocess_data(data_dir, train_dataset):
    train = pd.read_parquet(os.path.join(data_dir, train_dataset))
    train['event_time'] = pd.to_datetime(train['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    train = train.sort_values(by=['user_session', 'event_time'])

    train_df = train[['user_id', 'item_id', 'user_session', 'event_time']]
    train_df.loc[:, 'event_time'] = train_df['event_time'].values.astype(float)

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}

    with open(os.path.join(data_dir, 'user2idx.json'), "w") as f_user:
        json.dump(user2idx, f_user)
    with open(os.path.join(data_dir, 'item2idx.json'), "w") as f_item:
        json.dump(item2idx, f_item)

    train_df.loc[:, 'user_idx'] = train_df['user_id'].map(user2idx)
    train_df.loc[:, 'item_idx'] = train_df['item_id'].map(item2idx)

    train_df = train_df.dropna().reset_index(drop=True)
    train_df.rename(columns={
        'user_idx': 'user_idx:token', 
        'item_idx': 'item_idx:token', 
        'event_time': 'event_time:float'
    }, inplace=True)

    outdir = os.path.join(data_dir, 'SASRec_dataset')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_df[['user_idx:token', 'item_idx:token', 'event_time:float']].to_csv(
        os.path.join(outdir, 'SASRec_dataset.inter'), 
        sep='\t', 
        index=None
    )
    print('Recbole dataset generated')
    return train_df, user2idx, item2idx

class CustomTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.clip_grad_norm = {
            'max_norm': config['grad_clip_norm'] if 'grad_clip_norm' in config else 5.0,
            'norm_type': 2
        }

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                desc=f"Train {epoch_idx:>5}",
                ncols=100,
                postfix={"GPU RAM": f"{get_gpu_usage()} G/23.69 G"},
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            
            self._check_nan(loss)
            loss.backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.clip_grad_norm['max_norm'],
                    norm_type=self.clip_grad_norm['norm_type']
                )
            
            self.optimizer.step()
            
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(f"GPU RAM: {get_gpu_usage()} G/23.69 G")
        
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        print(f"Valid Result: {valid_result}")
        return valid_score, valid_result

def train_sasrec_model(data_dir, seed=42):
    set_seed(seed)

    config_dict = {
        'model': 'SASRec',
        'dataset': 'SASRec_dataset',
        'data_path': data_dir,
        'USER_ID_FIELD': 'user_idx',
        'ITEM_ID_FIELD': 'item_idx',
        'TIME_FIELD': 'event_time',
        'user_inter_num_interval': "[3,inf)",
        'item_inter_num_interval': "[3,inf)",
        'load_col': {
            'inter': ['user_idx', 'item_idx', 'event_time']
        },
        'n_layers': 3,
        'n_heads': 8,
        'hidden_dropout_prob': 0.3,
        'attn_dropout_prob': 0.2,
        'hidden_size': 256,
        'inner_size': 512,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'loss_type': 'BPR',
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'train_batch_size': 4096,
        'epochs': 100,
        'stopping_step': 10,
        'MAX_ITEM_LIST_LENGTH': 50,
        'grad_clip_norm': 5.0,  # 추가된 gradient clipping 설정
        'eval_args': {
            'split': {
                'LS': 'valid_and_test'
            },
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        },
        'metrics': ['Recall', 'NDCG', 'MRR', 'Precision'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        'eval_batch_size': 4096,
        'checkpoint_dir': './check_point',
        'device': 'cuda',
        'show_progress': True
    }

    config = Config(config_dict=config_dict)
    print('Config loaded')

    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, train_data.dataset).to(config['device'])
    print("Model information: ", model)

    trainer = CustomTrainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data, 
        saved=True, 
        show_progress=config["show_progress"]
    )

    print(f'Best Valid Result: {best_valid_result}')
    return model, dataset, test_data, config

def generate_recommendations(model, dataset, test_data, config, data_dir, output_dir, user2idx, item2idx):
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    train = pd.read_parquet(os.path.join(data_dir, train_dataset))
    train = train.sort_values(by=['user_session', 'event_time'])

    # item_idx 컬럼 추가
    train['item_idx'] = train['item_id'].map(item2idx)

    users = defaultdict(list)
    for u, i in zip(train['user_id'].map(user2idx), train['item_id'].map(item2idx)):
        users[u].append(i)

    # 인기 아이템 계산
    popular_items = train.groupby('item_id').size().sort_values(ascending=False)
    popular_top_10 = [item2idx[item_id] for item_id in popular_items.head(10).index]

    result = []
    
    for uid in tqdm(users, desc="Generating recommendations"):
        if str(uid) in dataset.field2token_id['user_idx']:
            recbole_id = dataset.token2id(dataset.uid_field, str(uid))
            topk_score, topk_iid_list = full_sort_topk(
                [recbole_id], model, test_data, k=10, device=config['device']
            )
            predicted_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            predicted_item_list = predicted_item_list[-1]
            predicted_item_list = list(map(int, predicted_item_list))
        else:
            predicted_item_list = popular_top_10

        for iid in predicted_item_list:
            result.append((idx2user[uid], idx2item[iid]))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    recommendations_df = pd.DataFrame(result, columns=["user_id", "item_id"])
    output_path = os.path.join(output_dir, "output.csv")
    recommendations_df.to_csv(output_path, index=False)
    print(f"Recommendations saved to {output_path}")

if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./output"
    train_dataset = "train.parquet"

    # 데이터 전처리
    train_df, user2idx, item2idx = preprocess_data(data_dir, train_dataset)
    
    # SASRec 모델 학습
    model, dataset, test_data, config = train_sasrec_model(data_dir)

    # 추천 생성 및 결과 저장
    generate_recommendations(model, dataset, test_data, config, data_dir, output_dir, user2idx, item2idx)