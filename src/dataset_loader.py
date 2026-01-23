import json
import torch
from torch.utils.data import Dataset
import config 

class DPRDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [수정] collate_fn에서 item['question'] 등으로 접근하므로 
        # 튜플이 아닌 원본 딕셔너리 객체를 그대로 반환해야 합니다.
        return self.data[idx]

def collate_fn(batch, q_tokenizer, ctx_tokenizer):
    """
    질문과 지문(정답+오답)을 인터리빙 방식으로 정렬하여 
    DPRNLLLoss의 target 인덱스와 수식적으로 동기화합니다.
    """
    queries = []
    passages = []
    
    for item in batch:
        # 1. 질문 추가: [Anchor [SEP] Path [CTX] Context] 구조
        queries.append(item['question'])
        
        # 2. 지문 정렬: [Positive, Hard_Negative_1, ..., Hard_Negative_M] 순서 유지
        # 이 순서는 loss_func.py의 passages_per_question 계산의 핵심 근거가 됩니다.
        
        # [Positive 지문 추가]
        pos_ctx = item['positive_ctxs'][0]
        # [수정] 데이터 빌더 로직과 일치하도록 제목과 본문을 결합 (구분자 없이 공백 권장)
        passages.append(f"{pos_ctx['title']} {pos_ctx['text']}")
        
        # [Hard Negative 지문 추가]
        # 이 루프를 통해 배치 내 지문 총합은 Batch_Size * (1 + num_hard_negs)가 됩니다.
        for hn_ctx in item.get('hard_negative_ctxs', []):
            passages.append(f"{hn_ctx['title']} {hn_ctx['text']}")

    # 3. 토크나이징 및 텐서 변환
    # [SEP]와 [CTX] 토큰은 train.py에서 등록했으므로 ID로 정상 변환됩니다.
    q_inputs = q_tokenizer(
        queries, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors='pt'
    )
    
    p_inputs = ctx_tokenizer(
        passages, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )

    return q_inputs, p_inputs