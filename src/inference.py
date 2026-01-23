import os
# [Windows 인코딩 패치]
os.environ["PYTHONUTF8"] = "1"

import torch
import torch.nn.functional as F
import spacy
import json
import re
import numpy as np
from transformers import AutoTokenizer, DPRQuestionEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher 
from spacy.symbols import ORTH 

# [사용자 정의 모듈 Import]
from dpr_architecture import SimpleDPRRetriever
from ss_architecture import SentenceRetrievalModel
from bert_architecture import MultiHopVerifier 

# =============================================================================
# 🚑 [Pyserini 인코딩 패치]
# =============================================================================
def patched_load_docids(self, docid_path):
    with open(docid_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f.readlines()]

FaissSearcher.load_docids = patched_load_docids
# =============================================================================


class SequentialRetriever:
    def __init__(self, dpr_path, faiss_path, wiki_path, ss_model_path, idf_path, verifier_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Spacy 설정
        if torch.cuda.is_available():
            try:
                spacy.require_gpu()
            except:
                pass
                
        try:
            self.nlp = spacy.load("en_core_web_trf")
            print("✅ Loaded Spacy TRF model.")
        except:
            self.nlp = spacy.load("en_core_web_sm")
            print("⚠️ TRF model not found. Using SM model.")
        self.nlp.tokenizer.add_special_case("gonna", [{ORTH: "gonna"}])
        
        # 2. IDF Table
        with open(idf_path, 'r', encoding='utf-8') as f:
            self.idf_table = json.load(f)
        self.dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

        # [PHASE 1] DPR Retrieval
        print(f"\n[Initializing Phase 1: DPR Retrieval]")
        try: self.dpr_model = SimpleDPRRetriever().to(self.device)
        except: pass
        
        if os.path.exists(dpr_path):
            print(f"   🔄 Overwriting Question Encoder weights")
            self.dpr_model.q_model = DPRQuestionEncoder.from_pretrained(dpr_path).to(self.device)
        self.dpr_model.eval()

        if os.path.exists(faiss_path):
            print(f"   📂 Loading FAISS Index")
            self.faiss_searcher = FaissSearcher(faiss_path, self.dpr_tokenizer.name_or_path)
        else:
            self.faiss_searcher = None
        
        self.doc_lookup = LuceneSearcher(wiki_path)

        # [PHASE 2] Sentence Selection (SS)
        print(f"\n[Initializing Phase 2: Sentence Selection]")
        self.sent_selector = SentenceRetrievalModel(model_path=ss_model_path, device=self.device.type)
        self.sent_selector.eval()

        # [PHASE 3] Verification (BERT)
        print(f"\n[Initializing Phase 3: Verification]")
        self.verifier_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.verifier_model = MultiHopVerifier(model_name="bert-base-cased", num_labels=3).to(self.device)
        if os.path.exists(verifier_path):
            try:
                self.verifier_model.load_state_dict(torch.load(verifier_path, map_location=self.device))
            except: pass
        self.verifier_model.eval()

    def get_idf_score(self, text):
        tokens = self.dpr_tokenizer.tokenize(text)
        return sum([self.idf_table.get(t, 1.0) for t in tokens])

    def _draw_bar(self, prob, width=20):
        filled = int(prob * width)
        return f"[{'#' * filled}{'-' * (width - filled)}] {prob*100:>5.1f}%"

    def dpr_search(self, query, k=3):
        if not self.faiss_searcher: return []
        inputs = self.dpr_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            q_emb = self.dpr_model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()
        hits = self.faiss_searcher.search(q_emb, k=k)
        
        results = []
        if hits and len(hits) > 0:
            target_hits = hits[0] if isinstance(hits[0], list) else hits
            for hit in target_hits:
                try:
                    doc = self.doc_lookup.doc(hit.docid)
                    if doc:
                        doc_json = json.loads(doc.raw())
                        results.append({
                            'docid': hit.docid,
                            'title': doc_json.get('title', str(hit.docid)),
                            'text': doc_json.get('text') or doc_json.get('contents') or "",
                            'score': hit.score
                        })
                except: continue
        return results

    # =========================================================================
    # [UPDATED] 우선순위 계산 함수
    # =========================================================================
    def _get_priority_score(self, span, is_entity=False):
        """
        2점: 고유 명사 (Entity)
        1점: 일반 명사 (Noun)
        0점: 대명사 (Pronoun)
        """
        if is_entity: return 2
        if span.root.pos_ in ["NOUN", "PROPN"]: return 1
        if span.root.pos_ == "PRON": return 0
        return -1

    def run_inference(self, data_json, k=3):
        claim = data_json['claim']
        gold_label = data_json['label'].upper() if data_json['label'] else "N/A"
        if gold_label == "SUPPORTED": gold_label = "SUPPORTS"
        
        gold_evidences = {}
        for sent in data_json.get('candidate_sentences', []):
            if sent.get('label') == 1:
                title = sent['title']
                if title not in gold_evidences: gold_evidences[title] = []
                gold_evidences[title].append(sent['text'])

        BOLD, RESET = "\033[1m", "\033[0m"
        GREEN, YELLOW = "\033[92m", "\033[93m"
        CYAN = "\033[96m"

        print(f"\n{BOLD}{'='*100}{RESET}")
        print(f"🔍 [PHASE 0] INPUT & GOLD STANDARD")
        print(f" 📢 CLAIM: {claim}")
        print(f" 🎯 GOLD LABEL: {GREEN}{gold_label}{RESET}")
        if gold_evidences:
            print(f" 📑 GOLD SENTENCES:")
            for title, sents in gold_evidences.items():
                for s in sents:
                    print(f"    {YELLOW}★{RESET} <{title}> {s[:100]}...")
        print(f"{BOLD}{'='*100}{RESET}")

        # ---------------------------------------------------------------------
        # [STEP 1] 키워드 추출
        # ---------------------------------------------------------------------
        doc = self.nlp(claim)
        candidates = []
        seen_texts = set()

        # 1. Entity
        for ent in doc.ents:
            if ent.text not in seen_texts:
                candidates.append({
                    "span": ent, 
                    "type": "ENTITY", 
                    "priority": self._get_priority_score(ent, is_entity=True)
                })
                seen_texts.add(ent.text)

        # 2. Noun / Pronoun
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text not in seen_texts:
                span = doc[token.i : token.i+1]
                p_score = self._get_priority_score(span)
                
                if p_score >= 0: 
                    k_type = "NOUN" if p_score == 1 else "PRON"
                    candidates.append({
                        "span": span, 
                        "type": k_type, 
                        "priority": p_score
                    })
                    seen_texts.add(token.text)

        # 스택 구성
        stack = []
        for item in candidates:
            span = item['span']
            head = span.root.head
            context_tokens = sorted([child for child in head.children] + [head], key=lambda t: t.i)
            context_phrase = " ".join([t.text for t in context_tokens])
            
            idf = self.get_idf_score(span.text)
            
            stack.append({
                "anchor": span.text,
                "context": context_phrase,
                "priority": item['priority'], 
                "idf": idf                   
            })

        # [정렬 로직] (Priority Asc -> IDF Asc)
        # pop()은 뒤에서부터 꺼내므로, 결국 높은 Priority & 높은 IDF 순으로 검색됨
        stack.sort(key=lambda x: (x['priority'], x['idf']))

        # =====================================================================
        # [NEW] 키워드 스택 시각화 (처리될 순서대로 출력)
        # =====================================================================
        print(f"\n{BOLD}📊 [KEYWORD STACK (Processing Order)]{RESET}")
        print(f"   {'#':<3} | {'Type':<8} | {'IDF':<6} | {'Keyword'}")
        print(f"   {'-'*3} | {'-'*8} | {'-'*6} | {'-'*30}")
        
        type_map = {2: "ENTITY", 1: "NOUN", 0: "PRON"}
        
        # stack은 뒤에서부터 꺼내지므로, 역순으로 출력해야 실제 처리 순서가 됨
        display_stack = stack[::-1]
        
        for i, item in enumerate(display_stack):
            p_label = type_map.get(item['priority'], "OTHER")
            print(f"   {i+1:<3} | {p_label:<8} | {item['idf']:<6.2f} | {item['anchor']}")
        
        print(f"   {'-'*55}")
        # =====================================================================

        seen_doc_ids = set()
        attempts = 0
        final_predicted_label = "N/A"
        is_correct = False
        
        while stack and attempts < 3:
            attempts += 1
            if not stack: break
            target = stack.pop() 
            
            # 로그 출력
            k_type = type_map.get(target['priority'], "OTHER")
            print(f"\n{BOLD}🚀 [ATTEMPT #{attempts}]{RESET} (Type: {k_type}, IDF: {target['idf']:.2f})")
            
            expanded_query = f"{target['anchor']} [SEP] {target['context']}"
            print(f" 📡 Query: \"{CYAN}{expanded_query}{RESET}\"")
            
            hits = self.dpr_search(expanded_query, k=k)
            
            if not hits:
                print(f"    ⚠️ No documents found by DPR.")
                continue

            all_sentences_pool, sentence_metadata = [], []
            for hit in hits:
                if hit['docid'] in seen_doc_ids: continue
                seen_doc_ids.add(hit['docid'])
                
                title = hit['title']
                content = hit['text']
                doc_sents = [s.text.strip() for s in self.nlp(content).sents if len(s.text.strip()) > 5]
                
                is_gold_doc = f"{YELLOW}★{RESET}" if title in gold_evidences else ""
                print(f"    📄 DPR Found: <{title}> {is_gold_doc} (Score: {hit['score']:.2f})")
                
                for s in doc_sents:
                    all_sentences_pool.append(s)
                    sentence_metadata.append({"title": title})

            if all_sentences_pool:
                results = self.sent_selector.retrieve(claim, all_sentences_pool, top_k=5)
                
                print(f"\n    {BOLD}📑 [PHASE 2] EVIDENCE SELECTION{RESET}")
                selected_evidences = []
                for s_idx, (p_idx, sim_score) in enumerate(results):
                    ev_sent = all_sentences_pool[p_idx]
                    src_title = sentence_metadata[p_idx]['title']
                    is_gold = False
                    if src_title in gold_evidences:
                        for gs in gold_evidences[src_title]:
                            if gs.strip() in ev_sent: is_gold = True; break
                    marker = f"{GREEN}✔{RESET}" if is_gold else ""
                    print(f"      [{s_idx+1}] {sim_score:.4f} {marker} <{src_title}> {ev_sent[:80]}...")
                    selected_evidences.append(ev_sent)

                combined_premise = " ".join(selected_evidences)
                inputs = self.verifier_tokenizer(combined_premise, claim, return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.no_grad():
                    logits = self.verifier_model(inputs['input_ids'], inputs['attention_mask'])
                    probs = F.softmax(logits, dim=-1).squeeze()
                
                p_s, p_r, p_n = probs[0].item(), probs[1].item(), probs[2].item()
                print(f"      • SUPPORTS: {self._draw_bar(p_s)} | REFUTES: {self._draw_bar(p_r)} | NEI: {self._draw_bar(p_n)}")

                if (p_n >= p_s + 0.3) and (p_n >= p_r + 0.3):
                    print(f"      {BOLD}Status:{RESET} ⚪ NEI Dominant. (Try next query)")
                elif abs(p_s - p_r) <= 0.3:
                    print(f"      {BOLD}Status:{RESET} 🟡 Ambiguous. (Try next query)")
                else:
                    final_predicted_label = "SUPPORTS" if p_s > p_r else "REFUTES"
                    is_correct = (final_predicted_label == gold_label)
                    print(f"      {BOLD}Status:{RESET} {'✅' if is_correct else '❌'} {final_predicted_label}")
                    if is_correct: break
        
        print(f"\n{BOLD}🏁 Final: {final_predicted_label} (Gold: {gold_label}){RESET}\n")

if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    IDF_PATH = os.path.join(BASE_PATH, "data", "idf_table_cache.json")
    
    DPR_Q_ENCODER_PATH = os.path.join(BASE_PATH, "saved_dpr_models", "dpr_finetuned", "best_model", "question_encoder")
    FAISS_INDEX_PATH = os.path.join(BASE_PATH, "hover_dpr_index")
    WIKI_INDEX_PATH = os.path.join(BASE_PATH, "hover_wiki_index")
    SS_MODEL_PATH = os.path.join(BASE_PATH, "saved_ss_models", "model.safetensors")
    VERIFIER_PATH = os.path.join(BASE_PATH, "saved_bert_models", "model.pth")

    engine = SequentialRetriever(
        dpr_path=DPR_Q_ENCODER_PATH,
        faiss_path=FAISS_INDEX_PATH,
        wiki_path=WIKI_INDEX_PATH,
        ss_model_path=SS_MODEL_PATH,
        idf_path=IDF_PATH,
        verifier_path=VERIFIER_PATH
    )
    
    test_json = {
        "uid": "1f6e406e-7d44-4d0f-a54d-766e5aa2e08c",
        "claim": "The Burlington Mall is a large, two-story, indoor shopping mall complex located off Route 3 and Route 128 in Burlington, Massachusetts.",
        "label": "SUPPORTED",
        "candidate_sentences": [
            {"title": "Burlington Mall (Massachusetts)", "sent_id": 1, "text": "The mall is one of many shopping venues in the Middlesex County area, including the Shops at Billerica, Square One Mall, Acton Plaza, and Middlesex Commons.", "label": 1},
            {"title": "Square One Mall", "sent_id": 0, "text": "Square One Mall (formerly the New England Shopping Center) is a 115 store shopping mall located along US Route 1 (Broadway) between Main Street and Essex Street in Saugus, Massachusetts.", "label": 1}
        ]
    }
    
    engine.run_inference(test_json)