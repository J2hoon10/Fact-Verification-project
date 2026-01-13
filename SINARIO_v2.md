# Project Design: Entity-Guided Iterative Retrieval & Selective Verification
**Date:** 2026-01-12
**Task:** Multi-hop Fact Verification (HOVER Dataset)
**Goal:** LLM 생성 없이, 데이터의 라벨(Evidence)을 단계별로 활용하여 효율적이고 설명 가능한 검증 모델 구축.

---

## 1. 개요 (Overview)

본 프로젝트는 HOVER 데이터셋의 복잡한 증거 구조를 **문서 레벨(Document-level)**과 **문장 레벨(Sentence-level)**로 분해하여 각기 다른 모델을 학습시키는 **파이프라인 접근 방식**을 채택한다.

검색 과정에서는 **엔티티(Entity)**의 확실성을 기준으로 순차적으로 정보를 탐색하며, 최종적으로 선별된 문장(Selected Sentences)만을 결합하여 추론 모델에 전달함으로써 연산 효율성과 정확도를 극대화한다.

---

## 2. 데이터 활용 및 학습 전략 (Data Utilization Strategy)

HOVER 데이터셋의 `evidence` 필드(`[[[Annotation_ID, Evidence_ID, Title, Sent_ID], ...]]`)를 분해하여 각 모듈의 정답지로 활용한다.

| 모듈 (Module) | 학습 목표 (Goal) | 활용 데이터 (Source) | 학습 방식 (Loss) |
| :--- | :--- | :--- | :--- |
| **1. Retriever** | 관련 문서 찾기 | `evidence[...][2]` (문서 제목) | **Contrastive Loss** (InfoNCE) |
| **2. Selector** | 핵심 문장 선별 | `evidence[...][3]` (문장 번호) | **BCE Loss** (Binary Classification) |
| **3. Reasoner** | 참/거짓 판별 | `label` (Supports/Refutes) | **Cross-Entropy Loss** (Multi-class) |

---

## 3. 모듈별 상세 설계 (Module Details)

### **Module 1: Query Analyzer (명제 분석기)**
* **Role:** Rule-based 전처리 및 검색 순서 결정.
* **Logic:**
    1.  **Normalization:** NFKD 정규화 적용 (예: `Elgström` → `Elgstrom`). **`[UNK]` 토큰 사용 금지.**
    2.  **Entity Parsing:** SpaCy 등을 활용해 명사 추출.
    3.  **Priority Queue:**
        * **Rank 1 (Anchor):** 확실한 고유명사 (Proper Noun).
        * **Rank 2 (Variable):** 대명사, 혹은 목적어가 생략된 관계절.

### **Module 2: Retriever (문서 검색기)**
* **Role:** 명제와 관련된 문서를 Top-K개 검색.
* **Architecture:** Bi-Encoder (Dense Retrieval).
* **Training Strategy:**
    * **Query:** 명제(Claim) 또는 확장된 쿼리.
    * **Positive:** `evidence`에 명시된 정답 문서.
    * **Negative:** BM25 점수는 높지만 정답이 아닌 문서 (Hard Negative) + 랜덤 문서.
    * **Objective:** 정답 문서와의 벡터 유사도를 최대화.

### **Module 3: Sentence Selector (문장 선택기) ⭐ 핵심**
* **Role:** 검색된 문서(약 20~50문장) 내에서 검증에 필수적인 문장(1~3문장)만 필터링.
* **Architecture:** Cross-Encoder (BERT-base).
* **Input Format:** `[CLS] Claim [SEP] Candidate Sentence [SEP]`
* **Training Strategy:**
    * **Positive (Label 1):** 정답 문서 내의 정답 문장 (`Sent_ID`).
    * **Negative (Label 0):** 정답 문서 내의 나머지 문장들 (In-doc Negative) + 오답 문서의 문장들.
    * **Robustness:** 오답 문서가 들어왔을 때 모든 문장을 0으로 예측하도록 학습하여 노이즈 내성 강화.

### **Module 4: Reasoner (추론기)**
* **Role:** 선별된 문장들을 근거로 최종 판결.
* **Architecture:** DeBERTa-v3-base (NLI Fine-tuning).
* **Input Format:** `[CLS] Claim [SEP] Selected Sentence 1 [SEP] Selected Sentence 2 ...`
* **Training Strategy:**
    * 학습 시에는 Retriever를 거치지 않고, **Gold Evidence(정답 문장)**를 직접 입력으로 주어 논리 학습에 집중시킴.

---

## 4. 추론 파이프라인 (Inference Pipeline)

테스트 시(Test-time)에는 위 모듈들이 유기적으로 연결되어 동작한다.

1.  **Step 1: 1차 검색 (Anchor Retrieval)**
    * 분석기가 추출한 **Anchor Entity**를 쿼리로 Retriever 실행.
    * 문서 2~3개 획득.

2.  **Step 2: 문장 선별 (Selection)**
    * 획득한 문서들의 모든 문장을 Selector에 입력.
    * **Score > 0.5** 인 문장만 추출 (`Evidence_Set A`).

3.  **Step 3: 반복 및 확장 (Iteration & Expansion)**
    * **Check:** `Evidence_Set A`로 Variable(대명사 등)이 해결되었는가? (간단한 NER 체크 or Score Threshold).
    * **Query Expansion:** 해결되지 않았다면, `원본 명제 + Evidence_Set A(요약)`을 쿼리로 2차 검색 실행.
    * 추가된 문서에서 다시 Selector 실행 $\rightarrow$ `Evidence_Set B` 획득.

4.  **Step 4: 최종 통합 및 판결 (Integration & Reasoning)**
    * 최종 입력: `[CLS] Claim [SEP] (Set A + Set B 결합) [SEP]`
    * Reasoner가 `Supports`, `Refutes`, `NEI` 중 하나로 분류.

---

## 5. 핵심 구현 포인트 (Key Implementation Points)

1.  **데이터 파싱 (Data Parsing):**
    * HOVER `evidence` 필드는 3중 리스트 구조임.
    * 학습 데이터 구축 시, 가장 안쪽 리스트의 `Index 2(Title)`은 Retriever용, `Index 3(Sent_ID)`는 Selector용으로 정확히 분리해서 사용해야 함.
    
2.  **알 수 없는 명사 처리:**
    * `[UNK]` 토큰 치환 절대 금지.
    * `Elgström` 같은 단어는 Subword Tokenizer가 처리하도록 원문 유지.
    * 목적어가 없는 경우(Missing Object)에는 `[MASK]` 토큰을 활용하거나 1차 검색 결과를 문맥으로 채워서 검색.

3.  **성능 최적화:**
    * Selector는 연산량이 많으므로(Cross-Encoder), Retriever가 가져오는 문서를 Top-3~5개로 제한하거나, 문장이 너무 많은 문서는 앞부분만 자르는 등의 전처리가 필요할 수 있음.