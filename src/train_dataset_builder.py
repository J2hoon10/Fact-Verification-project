import os
import json
import spacy
import torch
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyserini.search.lucene import LuceneSearcher
from spacy.symbols import ORTH

# [1] 환경 및 리소스 설정
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-25"
os.environ["PYTHONUTF8"] = "1"

# GPU 사용 가능 여부 확인 (우선순위 설정)
USE_GPU = torch.cuda.is_available()
NUM_CORES = 6  # CPU 모드 시 사용할 코어 제한

# 글로벌 변수
searcher = None
nlp = None

def load_nlp_model():
    """상황에 맞는 spaCy 모델 로드"""
    if USE_GPU:
        # GPU 모드: 단일 프로세스에서 GPU 가속 활용
        spacy.require_gpu()
        print("🚀 [GPU Mode] Prioritizing GPU acceleration for Transformer models.")
    else:
        # CPU 모드: 멀티프로세싱 활용 예정
        print(f"🚀 [CPU Mode] Falling back to CPU with {NUM_CORES} cores.")
        
    try:
        model = spacy.load("en_core_web_trf")
    except:
        model = spacy.load("en_core_web_sm")
    model.tokenizer.add_special_case("gonna", [{ORTH: "gonna"}])
    return model

def init_worker(index_path):
    """CPU 병렬 프로세스용 초기화 함수"""
    global searcher, nlp
    searcher = LuceneSearcher(index_path)
    # CPU 모드에서는 개별 프로세스가 모델을 로드함
    if not USE_GPU:
        nlp = load_nlp_model()

def get_passage_text(title, current_searcher):
    """Lucene 인덱스에서 본문 조회"""
    hits = current_searcher.search(f"title:\"{title}\"", k=1)
    if hits:
        doc = current_searcher.doc(hits[0].docid)
        return json.loads(doc.raw()).get('text') or json.loads(doc.raw()).get('contents') or ""
    return ""

def get_logical_path(gold_titles, current_searcher):
    """[로직] 브릿지 엔티티 기반 논리적 경로 재정렬"""
    if len(gold_titles) <= 1: return gold_titles
    title_to_text = {t: get_passage_text(t, current_searcher).lower() for t in gold_titles}
    temp_titles = gold_titles.copy()
    for i in range(len(temp_titles)):
        for j in range(len(temp_titles)):
            if i == j: continue
            p, c = temp_titles[i], temp_titles[j]
            if c.lower() in title_to_text.get(p, ""):
                idx_p, idx_c = temp_titles.index(p), temp_titles.index(c)
                if idx_p > idx_c:
                    temp_titles[idx_p], temp_titles[idx_c] = temp_titles[idx_c], temp_titles[idx_p]
    return temp_titles

def mine_hard_negatives(query_text, positive_titles, current_searcher, k=1):
    """[로직] 쿼리별 동적 Hard Negative Mining"""
    hits = current_searcher.search(query_text, k=20) 
    hard_negs = []
    for hit in hits:
        if hit.docid in positive_titles: continue
        doc = current_searcher.doc(hit.docid)
        if doc:
            doc_json = json.loads(doc.raw())
            title = doc_json.get('title', hit.docid)
            if title not in positive_titles:
                hard_negs.append({
                    "title": title,
                    "text": doc_json.get('text') or doc_json.get('contents') or "",
                    "score": hit.score
                })
        if len(hard_negs) >= k: break
    return hard_negs

def process_item(item, current_nlp, current_searcher):
    """단일 데이터 가공 핵심 로직"""
    if item['label'] != 'SUPPORTED': return []

    claim = item['claim']
    gold_titles = list(set([fact[0] for fact in item['supporting_facts']]))
    logical_titles = get_logical_path(gold_titles, current_searcher)
    
    # Gold 문서 본문 사전 로드 (문장 분절 포함)
    gold_texts = {t: [s.text for s in current_nlp(get_passage_text(t, current_searcher)).sents] for t in gold_titles}
    
    doc = current_nlp(claim)
    candidates = []
    seen = set()
    HIGH_PRIORITY = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}
    
    for ent in doc.ents:
        if ent.label_ in HIGH_PRIORITY:
            candidates.append(ent)
            seen.add(ent.text)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "PRON"] and token.text not in seen and not token.is_stop:
            candidates.append(doc[token.i : token.i + 1])
            seen.add(token.text)

    entries = []
    for span in candidates:
        anchor = span.text
        matched = [t for t in gold_titles if anchor.lower() in t.lower() or t.lower() in anchor.lower()]
        if not matched: continue
        best_pos_title = matched[0]

        # [로직] 순서 보존형 비대칭 슬라이싱
        # $$Slicing\_Context = doc[\max(0, span.start - 2) : \min(len(doc), span.end + 6)]$$
        start_i, end_i = max(0, span.start - 2), min(len(doc), span.end + 6)
        slicing_context = doc[start_i:end_i].text
        
        target_idx = logical_titles.index(best_pos_title)
        path_str = " -> ".join(["Claim"] + logical_titles[:target_idx])
        query_text = f"{anchor} [SEP] {path_str} [CTX] {slicing_context}"
        
        hard_negs = mine_hard_negatives(query_text, gold_titles, current_searcher, k=1)

        entries.append({
            "question": query_text,
            "positive_ctxs": [{"title": best_pos_title, "text": " ".join(gold_texts[best_pos_title])}],
            "negative_ctxs": [],
            "hard_negative_ctxs": hard_negs
        })
    return entries

# CPU용 매핑 함수
def cpu_worker(item):
    return process_item(item, nlp, searcher)

def build_dataset(input_path, output_path, index_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    final_results = []

    if USE_GPU:
        # [GPU 모드] 병렬 처리 없이 단일 루프에서 GPU 가속 사용
        print("🚀 Starting Dataset Generation with GPU...")
        nlp_model = load_nlp_model()
        local_searcher = LuceneSearcher(index_path)
        
        for item in tqdm(data, desc="Processing (GPU)"):
            final_results.extend(process_item(item, nlp_model, local_searcher))
    else:
        # [CPU 모드] 6개 코어 제한 병렬 처리
        print(f"🚀 Starting Dataset Generation with CPU (Cores: {NUM_CORES})...")
        with Pool(processes=NUM_CORES, initializer=init_worker, initargs=(index_path,)) as pool:
            for result_list in tqdm(pool.imap_unordered(cpu_worker, data), total=len(data), desc="Processing (CPU)"):
                final_results.extend(result_list)

    print(f"✅ Saving {len(final_results)} entries to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    IN_FILE = "./data/hover/hover_dev_release_v1.1.json"
    OUT_FILE = "./data/dpr_train_data/dpr_dev.json"
    IDX_PATH = "./hover_wiki_index"
    
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    build_dataset(IN_FILE, OUT_FILE, IDX_PATH)