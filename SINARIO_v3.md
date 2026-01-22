# SINARIO v3: Attention-Guided Iterative Verification Pipeline

**Date:** 2026-01-22
**Version:** v3.1 (Code-Aligned)
**Scenario Target:** Multi-hop Claim Verification (HOVER) without LLM Generation

---

## 🎯 0. 시나리오 개요 (Scenario Overview)

**Target Claim:**
> *"The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$."*

이 시나리오는 위 명제가 입력되었을 때, 모델이 **Spacy 구문 분석을 통해 키워드 스택(Stack)을 쌓고, 우선순위가 높은 순서대로 독립적인 검색을 수행하여 증거를 수집하는** 단계별 내부 연산 과정을 상세히 기술한다.

---

## 🏗️ Phase 1: Preprocessing & Stack Construction (전처리 및 스택 생성)

### 1.1 Linguistic Analysis (구문 분석)
명제가 입력되면 **Spacy NLP Pipeline(`en_core_web_trf`)**을 실행하여 품사(POS)와 구문 구조를 분석한다.

* **Entity & POS Tagging:**
    * `doc.ents`: 고유명사(Entity) 식별 (예: `M.I.L.F.$`, `Fergie`, `Polow da Don`, `Life Goes On`).
    * `Token.pos_`: 불용어(Stopword)를 제외한 `NOUN`, `PRON` 식별.
* **Syntactic Context Extraction:**
    * 각 키워드(Anchor)의 문맥을 파악하기 위해, 지배소(Head)와 그 자식 노드(Children)들을 묶어 **Local Context**를 추출한다.
    * $$C_{local}(w) = \text{Head}(w) \cup \text{Children}(\text{Head}(w))$$

### 1.2 Priority Logic (우선순위 산정)
키워드의 검색 순서는 **범주형 점수(Category Score)**를 최우선으로 하고, 동점일 경우 **IDF(희소성)**로 결정한다.

**[Priority Scoring Rule]**
1.  **Level 2 (Entity):** 고유명사 (가장 중요)
2.  **Level 1 (Noun):** 일반 명사
3.  **Level 0 (Pronoun):** 대명사

### 1.3 Keyword Stack Construction (스택 생성)
추출된 키워드들은 리스트에 담긴 후 `(Priority, IDF)` 기준으로 정렬된다. `pop()` 연산을 통해 뒤에서부터 꺼내지므로, **[Entity & High IDF]**가 가장 먼저 실행된다.

| Order (Pop) | Keyword ($k$) | Type (Priority) | IDF | Context ($C_{local}$) |
| :---: | :--- | :---: | :---: | :--- |
| **1 (Start)** | **`M.I.L.F.$`** | **Entity (2)** | **9.5** | *was M.I.L.F.$* |
| 2 | `Polow da Don` | Entity (2) | 8.0 | *produced by Polow da Don* |
| 3 | `Life Goes On` | Entity (2) | 6.5 | *followed by Life Goes On* |
| 4 | `Fergie` | Entity (2) | 4.0 | *recorded by Fergie* |
| 5 | `song` | Noun (1) | 3.2 | *The song recorded* |

---

## 🔍 Phase 2: 1st Retrieval & Selection (1차 검색 및 선별)

### 2.1 Contextual Query Formulation (쿼리 생성)
Stack에서 가장 높은 순위인 **`M.I.L.F.$`**를 꺼내어(Pop), Phase 1.1에서 추출해둔 구문적 문맥(Context)과 결합한다.

* **Logic:** $Q_1 = \text{Anchor} \oplus \texttt{[SEP]} \oplus C_{local}$
* **Query:** `M.I.L.F.$ [SEP] was M.I.L.F.$`

### 2.2 Dense Retrieval (DPR Search)
Bi-Encoder($E_Q, E_D$)를 통해 유사도가 높은 문서를 검색한다.

* **Retrieved Candidates:**
    * Doc A (`M.I.L.F.$`): Score **High**
    * Doc B (`Fergie Discography`): Score **Mid**

### 2.3 Sentence Selection (핵심 문장 압축)
Selector(Cross-Encoder)가 검색된 문서들의 문장($S_i$)에 대해 증거 확률을 계산한다.

* **Input:** `[CLS] Claim [SEP] Doc_Sentence_i`
* **Selected Evidence ($E_1$):**
    * *"M.I.L.F.$ is a song by Fergie... produced by Polow da Don."* (Confidence High)

---

## 🚦 Phase 3: Gatekeeper Verification (연결성 검증)

### 3.1 NLI Inference (논리 검증)
선별된 증거($E_1$)만으로 명제 전체($C$)를 검증할 수 있는지 **BERT Verifier**가 판단한다.

* **Input:** `[CLS] Claim [SEP] E_1 (M.I.L.F.$ is a song... Polow da Don) [SEP]`
* **Model Output Probabilities:**
    * `Supports`: **0.60**
    * `Refutes`: 0.05
    * `NEI`: **0.35**

### 3.2 Decision Logic (분기 처리)
코드에 구현된 로직에 따라 상태를 판별한다.

* **Condition:** `(NEI >= Supports + 0.3)` OR `(abs(Supports - Refutes) <= 0.3)`
* **Current State:** Supports(0.60)가 가장 높지만, NEI(0.35)와의 차이가 크지 않거나 확실한 임계값을 넘지 못해 **불확실(Ambiguous)** 상태로 판단될 수 있음.
* **Action:** **CONTINUE (Try Next Stack Item)**
    * *Reasoning:* Doc A는 `Polow da Don` 정보는 확인해주었으나, `Life Goes On`에 대한 정보가 부족함.

---

## 🔄 Phase 4: Iteration & Expansion (반복 및 확장)

### 4.1 Next Keyword Selection (다음 키워드 선정)
Stack의 다음 순위 키워드를 Pop한다.
* **Target:** **`Life Goes On`** (Entity, IDF 6.5)
* **Context:** `followed by Life Goes On` (Head: `followed`, Children: `by`, `Life`, `Goes`, `On`)

### 4.2 Independent Query Expansion (독립 쿼리 확장)
이전 문서의 내용을 쿼리에 섞지 않고, **새로운 키워드에 집중하여** 독립적인 검색을 수행한다.

* **Formula:** $Q_2 = \text{Target} \oplus \texttt{[SEP]} \oplus C_{local}(\text{Target})$
* **Query:** `Life Goes On [SEP] followed by Life Goes On`

### 4.3 2nd Retrieval & Selection
* **Retrieved:** Doc C (`Life Goes On (song)`)
* **Selected Sentence ($E_2$):**
    * *"Life Goes On is a song by Fergie... released as the second single from Double Dutchess, following M.I.L.F.$."*

---

## ⚖️ Phase 5: Final Reasoning (최종 판결)

### 5.1 Evidence Integration (증거 통합)
1차($E_1$) 및 2차($E_2$) 검색에서 얻은 모든 증거 문장을 하나로 연결한다.
$$E_{final} = \text{join}(E_1, E_2)$$

### 5.2 Final NLI Classification
* **Input:** `[CLS] Claim [SEP] E_final [SEP]`
    * *Claim:* "...produced by Polow da Don and followed by Life Goes On..."
    * *Evidence:* "...produced by Polow da Don..." ($E_1$) + "...following M.I.L.F.$..." ($E_2$)
* **Result:**
    * `Supports`: **0.95** (Dominant)
    * `Refutes`: 0.02
    * `NEI`: 0.03

### 5.3 Verdict
**Label:** **SUPPORTS (참)** (Gold Label과 일치하므로 루프 종료)

---

## 🔁 6. Feedback & Backtracking Logic (피드백 요약)

시스템이 정답을 확신할 때까지 스택을 소비하며 순환하는 로직을 시각화한다.

### 6.1 Logic Flowchart
```mermaid
graph TD
    Start([Start]) --> A[Phase 1: Spacy 분석 & Stack 생성]
    A --> B{Stack Empty OR Attempts >= 3?}
    B -- Yes --> Finish(["최종 판결 (Final Output)"])
    B -- No --> C[Pop Target Keyword]
    
    C --> D[Phase 2: Query(Keyword + SEP + Context) 생성]
    D --> E[DPR 검색 & Evidence Selection]
    
    E --> F[Evidence Pool 업데이트 (E_total)]
    F --> G[Phase 3: Verifier (NLI) 수행]
    
    G --> H{Is Clear Decision?}
    H -- "Yes (High Confidence)" --> Stop(["Stop & Return Label"])
    H -- "No (Ambiguous / NEI)" --> I[Log Status & Continue]
    I --> B
