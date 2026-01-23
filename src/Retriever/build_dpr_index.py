import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import DPRContextEncoder, AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

# =========================================================
# [설정] 본인 환경에 맞게 수정하세요
# =========================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 1. 학습된 Context Encoder 경로
CTX_ENCODER_PATH = os.path.join(BASE_PATH, "saved_dpr_models", "dpr_finetuned", "best_model", "ctx_encoder")

# 2. Lucene 인덱스 경로
WIKI_INDEX_PATH = os.path.join(BASE_PATH, "hover_wiki_index")

# 3. 결과물 저장 경로
OUTPUT_FAISS_PATH = os.path.join(BASE_PATH, "hover_dpr_index")

# [중요] 속도 향상을 위해 배치 사이즈를 늘림 (VRAM 16GB 기준 128~256 추천)
# OOM(Out of Memory) 에러가 나면 64로 줄이세요.
BATCH_SIZE = 1536
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================

def build_index():
    print(f"🚀 [Start] Building FAISS Index (Fast Mode - FP16)...")
    print(f"   - Context Encoder: {CTX_ENCODER_PATH}")
    print(f"   - Batch Size     : {BATCH_SIZE}")
    print(f"   - Device         : {DEVICE}")

    # 1. 모델 & 토크나이저 로드 (FP16 적용)
    try:
        # .half()를 호출하여 모델을 FP16 모드로 전환 (속도 2배 향상)
        ctx_encoder = DPRContextEncoder.from_pretrained(CTX_ENCODER_PATH).to(DEVICE).half()
        ctx_encoder.eval()
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        print("✅ Context Encoder loaded in FP16 mode.")
    except Exception as e:
        print(f"❌ Failed to load Context Encoder: {e}")
        return

    # 2. Lucene Index 로드
    if not os.path.exists(WIKI_INDEX_PATH):
        print(f"❌ Lucene index not found at {WIKI_INDEX_PATH}")
        return
    
    searcher = LuceneSearcher(WIKI_INDEX_PATH)
    num_docs = searcher.num_docs
    print(f"✅ Lucene Index loaded. Total documents: {num_docs}")

    # 3. FAISS 인덱스 초기화
    d = 768
    index = faiss.IndexFlatIP(d) # Inner Product (Cosine Sim)

    os.makedirs(OUTPUT_FAISS_PATH, exist_ok=True)

    # 4. 인코딩 루프
    doc_ids = []
    batch_texts = []
    batch_ids = []

    print("🔄 Encoding documents...")
    
    # torch.no_grad()로 그래디언트 계산 방지 (메모리 절약)
    with torch.no_grad():
        for i in tqdm(range(num_docs), desc="Indexing"):
            try:
                # Lucene에서 문서 가져오기
                doc = searcher.doc(i)
                if doc is None: continue
                
                raw_json = json.loads(doc.raw())
                
                # ID 및 텍스트 추출
                d_id = raw_json.get('id') or raw_json.get('_id') or str(raw_json.get('title'))
                title = raw_json.get('title', "")
                text = raw_json.get('text') or raw_json.get('contents') or ""
                
                # DPR 입력 포맷: "Title [SEP] Text"
                full_text = f"{title} [SEP] {text}"
                
                batch_texts.append(full_text)
                batch_ids.append(d_id)

                # 배치가 꽉 찼을 때 인코딩 수행
                if len(batch_texts) >= BATCH_SIZE:
                    # 토크나이징
                    inputs = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=256
                    ).to(DEVICE)
                    
                    # [핵심] FP16 연산 수행
                    # autocast를 쓰거나 모델이 이미 .half() 상태이므로 바로 forward
                    outputs = ctx_encoder(**inputs)
                    
                    # FP16 결과를 다시 FP32(float32)로 변환 (FAISS는 float32 선호)
                    embeddings = outputs.pooler_output.float().cpu().numpy()
                    
                    # FAISS에 추가
                    index.add(embeddings)
                    doc_ids.extend(batch_ids)
                    
                    # 리셋
                    batch_texts = []
                    batch_ids = []

            except Exception as e:
                continue

        # 남은 자투리 배치 처리
        if batch_texts:
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to(DEVICE)
            
            outputs = ctx_encoder(**inputs)
            embeddings = outputs.pooler_output.float().cpu().numpy()
            index.add(embeddings)
            doc_ids.extend(batch_ids)

    print(f"✅ Encoding finished. Total vectors: {index.ntotal}")

    # 5. 저장
    faiss_file = os.path.join(OUTPUT_FAISS_PATH, "index")
    faiss.write_index(index, faiss_file)
    print(f"💾 Saved FAISS index to {faiss_file}")

    docid_file = os.path.join(OUTPUT_FAISS_PATH, "docid")
    with open(docid_file, 'w', encoding='utf-8') as f:
        for did in doc_ids:
            f.write(f"{did}\n")
    print(f"💾 Saved DocID mapping to {docid_file}")

if __name__ == "__main__":
    # CUDA 캐시 비우기 (메모리 확보)
    torch.cuda.empty_cache()
    build_index()