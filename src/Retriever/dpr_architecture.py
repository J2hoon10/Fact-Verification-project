import torch.nn as nn
from transformers import DPRQuestionEncoder, DPRContextEncoder as HF_DPRContextEncoder
from loss_func import DPRNLLLoss
import config

class SimpleDPRRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_model = DPRQuestionEncoder.from_pretrained(config.Config.Q_ENCODER_ID)

    def forward(self, input_ids, attention_mask):
        # pooler_output을 사용하여 문장 전체의 벡터 표현(768 dim)을 얻음
        return self.q_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

class DPRContextEncoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_model = HF_DPRContextEncoder.from_pretrained(config.Config.CTX_ENCODER_ID)

    def forward(self, input_ids, attention_mask):
        return self.ctx_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

class DPRBiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.question_encoder = SimpleDPRRetriever()
        self.ctx_encoder = DPRContextEncoderWrapper()
        self.loss_fn = DPRNLLLoss()
        
    def forward(self, q_input_ids, q_attention_mask, p_input_ids, p_attention_mask):
        q_vectors = self.question_encoder(q_input_ids, q_attention_mask)
        p_vectors = self.ctx_encoder(p_input_ids, p_attention_mask)
        # NLL Loss와 유사도 점수 반환
        loss, scores = self.loss_fn(q_vectors, p_vectors)
        return loss

    # [중요] 추가된 토큰 수에 맞춰 임베딩 레이어 크기를 조정하는 메서드
    def resize_token_embeddings(self, q_vocab_size, ctx_vocab_size):
        self.question_encoder.q_model.resize_token_embeddings(q_vocab_size)
        self.ctx_encoder.ctx_model.resize_token_embeddings(ctx_vocab_size)