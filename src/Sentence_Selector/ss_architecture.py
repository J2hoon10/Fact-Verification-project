"""
ss_architecture.py
Sentence Retrieval 모델
Bi-encoder 방식으로 Claim과 Sentence 간의 유사도를 계산하여 관련 문장을 검색
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from safetensors.torch import load_file

class SentenceRetrievalModel(nn.Module):
    """
    Sentence Retrieval 모델 (Bi-encoder 방식)
    Claim과 Sentence를 각각 인코딩하여 유사도를 계산합니다.
    """
    
    def __init__(
        self,
        model_path: str = "sentence-transformers/all-MiniLM-L6-v2", # [수정] model_name -> model_path 변경
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: 로컬 모델 경로(.safetensors) 또는 HuggingFace 모델명
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        super(SentenceRetrievalModel, self).__init__()
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔄 [SS Model] Loading Sentence Retrieval Model from: {model_path}")
        
        # ---------------------------------------------------------------------
        # [로딩 로직 개선] 로컬 파일(.safetensors)인 경우 처리
        # ---------------------------------------------------------------------
        # 1. 단일 가중치 파일이 지정된 경우 (config.json 없음)
        if os.path.isfile(model_path):
            print(f"   ⚠️ Detected local weight file. Initializing base architecture first.")
            # 껍데기(아키텍처)는 기본 모델에서 가져옵니다.
            base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.encoder = SentenceTransformer(base_model_name, device=self.device)
            
            # 가중치 덮어씌우기
            print(f"   📂 Loading weights from: {model_path}")
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location=self.device)
            
            # 'encoder.' 접두사가 있다면 제거하고 로드
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    new_state_dict[k.replace("encoder.", "")] = v
                else:
                    new_state_dict[k] = v
            
            self.encoder.load_state_dict(new_state_dict, strict=False)
            print("   ✅ Local weights loaded successfully.")
            
        # 2. 일반적인 경우 (HuggingFace 이름 또는 폴더 경로)
        else:
            self.encoder = SentenceTransformer(model_path, device=self.device)
            
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"   - Device: {self.device}")
        print(f"   - Embedding Dim: {self.embedding_dim}")
    
    def encode_claim(self, claims: List[str], requires_grad: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if not requires_grad:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder.encode(claims, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        else:
            self.encoder.train()
            tokenizer = self.encoder.tokenizer
            model = self.encoder[0].auto_model
            
            encoded = tokenizer(claims, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            model_output = model(**encoded)
            embeddings = model_output[0]
            input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    def encode_sentences(self, sentences: List[str], requires_grad: bool = False) -> Union[np.ndarray, torch.Tensor]:
        # 문장 인코딩은 Claim 인코딩과 동일한 방식 사용
        return self.encode_claim(sentences, requires_grad)

    def retrieve(self, claim: str, candidate_sentences: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if not candidate_sentences:
            return []
            
        # 인코딩
        claim_emb = self.encode_claim([claim])[0]
        sent_embs = self.encode_sentences(candidate_sentences)
        
        # 유사도 계산
        similarities = cosine_similarity([claim_emb], sent_embs)[0]
        
        # Top-K 선택
        top_k = min(top_k, len(candidate_sentences))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results