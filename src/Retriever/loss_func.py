import torch
import torch.nn as nn
import torch.nn.functional as F

class DPRNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_vectors, p_vectors):
        """
        In-batch Negative를 활용한 NLL Loss 계산
        q_vectors: (B, D)
        p_vectors: (B * (1 + num_hard_negs), D)
        """
        # 1. 유사도 행렬 계산 (Batch 내 모든 질문과 모든 지문 간의 Dot Product)
        scores = torch.matmul(q_vectors, p_vectors.transpose(0, 1)) # (B, B*(1+M))
        
        batch_size = q_vectors.size(0)
        num_passages = p_vectors.size(0)
        
        if batch_size == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True), scores

        # 2. 정답 타겟 인덱스 생성
        # 각 질문에 대해 매칭되는 긍정 지문의 위치를 찾음
        passages_per_question = num_passages // batch_size 
        
        # 정답 인덱스는 [0, M+1, 2*(M+1), ...] 형식이 됨
        target = torch.arange(
            0, num_passages, step=passages_per_question, 
            device=scores.device, dtype=torch.long
        )
        
        # 3. NLL Loss 계산 (Cross Entropy)
        loss = F.cross_entropy(scores, target)
        
        return loss, scores