# SINARIO v3: Attention-Guided Iterative Verification Pipeline

**Date:** 2026-01-13
**Version:** Final Release (Optimized)
**Scenario Target:** Multi-hop Claim Verification (HOVER) without LLM Generation

---

## 🎯 0. 시나리오 개요 (Scenario Overview)

**Target Claim:**
> *"The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$."*

이 시나리오는 위 명제가 입력되었을 때, 모델이 **어떻게 단어를 분석하고, 문서를 찾고, 부족한 정보를 인식하여 최종 판결을 내리는지** 단계별 내부 연산 과정을 상세히 기술한다.

---

## 🏗️ Phase 1: Preprocessing & Analysis (전처리 및 분석)

### 1.1 Offline IDF Calculation (사전 연산)
위키피디아 전체 문서($D_{wiki}$)를 대상으로 모든 단어($w$)의 IDF 값을 미리 계산하여 테이블(`Hash Map`)로 저장해둔다.
$$\text{IDF}(w) = \log \left( \frac{|D_{wiki}|}{df(w) + 1} \right)$$

### 1.2 Claim Analysis (통합 언어 분석)
명제가 입력되면 **SpaCy 파이프라인**과 **BERT Query Encoder**를 한 번씩만 실행하여 필요한 모든 언어적 정보를 추출하고 캐싱(Caching)한다. **(Compute Once, Use Everywhere)**

* **Linguistic Analysis (SpaCy):**
    * `nlp(Claim)`을 실행하여 `Doc` 객체 생성.
    * **POS Tagging:** 고유명사(`M.I.L.F.$` 등), 명사, 동사 식별.
    * **Dependency Parsing:** 단어 간의 수식 관계(Tree 구조)를 미리 파악하여 메모리에 저장. (Phase 2에서 재사용)
* **Attention Extraction (BERT):**
    * Query Encoder의 Last Layer에서 `[CLS]` 토큰의 Attention Weight 추출.
    * $$\text{Attn}(w) = \frac{1}{H} \sum_{h=1}^{H} A_{h, last}[0, w_{idx}]$$

### 1.3 Priority Queue Construction (우선순위 큐 생성)
각 키워드($k$)에 대해 **Hybrid Priority Score**를 계산하여 정렬한다.

$$\text{Score}(k) = \text{IDF}(k) \times (1 + \text{Attn}(k))$$

| Rank | Keyword ($k$) | IDF (희소성) | Attn (문맥 중요도) | **Total Score** | 비고 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **`M.I.L.F.$`** | 9.5 (Very High) | 0.25 (High) | **11.87** | **Anchor (선정)** |
| 2 | `Polow da Don` | 8.0 (High) | 0.15 (Med) | 9.20 | Queue 대기 |
| 3 | `Life Goes On` | 6.5 (Med) | 0.10 (Low) | 7.15 | Queue 대기 |
| 4 | `Fergie` | 4.0 (Low) | 0.22 (High) | 4.88 | Queue 대기 |

---

## 🔍 Phase 2: 1st Retrieval & Selection (1차 검색 및 선별)

### 2.1 Contextual Query Formulation (쿼리 생성)
Phase 1.2에서 생성해둔 **SpaCy Doc 객체(의존 구문 트리)**를 즉시 조회하여, 별도의 연산 없이 Anchor 키워드와 연결된 수식어를 추출한다.

* **Lookup:** `Doc` 트리에서 Anchor(`M.I.L.F.$`)의 Head(`was`)와 연결된 종속절(`song recorded by Fergie`) 탐색.
* **Formula:** $Q_1 = \text{Anchor} \oplus \texttt{[SEP]} \oplus \text{Pre-computed Context}$
* **Result:** `M.I.L.F.$ [SEP] song recorded by Fergie`

### 2.2 Dense Retrieval & Filtering (검색 및 필터링)
Bi-Encoder($E_Q, E_D$)를 통해 유사도($Sim$)가 높은 문서를 검색한다.

$$Sim(Q_1, D) = E_Q(Q_1) \cdot E_D(D)^T$$

* **Retrieved Candidates:**
    * Doc A (`M.I.L.F.$`): Score **0.88** (Threshold 0.5 초과 $\rightarrow$ **Pass**)
    * Doc B (`Double Dutchess`): Score 0.45 (Fail $\rightarrow$ Drop)
    * ...

### 2.3 Sentence Selection (핵심 문장 압축)
Selector(Cross-Encoder)가 Doc A의 모든 문장($S_i$)에 대해 증거 확률을 계산한다.

* **Input:** `[CLS] Claim [SEP] Doc_A_Sentence_i`
* **Output:**
    * $S_1$: *"M.I.L.F.$ is a song by Fergie... produced by Polow da Don."* ($P=0.98$)
    * $S_2$: *"It was released as a single..."* ($P=0.12$)
* **Selected Set ($E_1$):** `[S_1]`

---

## 🚦 Phase 3: Gatekeeper Verification (연결성 검증)

### 3.1 NLI Inference (논리 검증)
선별된 문장($E_1$)만으로 명제 전체($C$)를 검증할 수 있는지 **NLI 모델**이 판단한다.

* **Input:** `[CLS] Claim [SEP] E_1 (M.I.L.F.$ is a song... Polow da Don) [SEP]`
* **Model Output Probabilities:**
    * `Entailment`: **0.65**
    * `Neutral`: **0.34**
    * `Contradiction`: 0.01

### 3.2 Decision Logic (분기 처리)
$$\text{Decision} = \begin{cases} \text{STOP (Final)} & \text{if } P(Ent) > 0.9 \\ \text{ITERATE (Bridge)} & \text{if } 0.3 < P(Ent) \le 0.9 \\ \text{REJECT (Backtrack)} & \text{if } P(Ent) \le 0.3 \end{cases}$$

* **Current State:** $P(Ent) = 0.65$ $\rightarrow$ **Zone B: ITERATE (Bridge)**
* **Reasoning:** Doc A는 `Fergie`와 `Polow da Don` 정보를 포함하고 있어 관련성은 높으나, `Life Goes On`에 대한 정보가 결여됨. 따라서 **추가 탐색** 결정.

---

## 🔄 Phase 4: Iteration & Expansion (반복 및 확장)

### 4.1 Next Keyword Selection (다음 키워드 선정)
Queue의 다음 순위 키워드를 확인한다.
* Rank 2: `Polow da Don` (이미 Doc A에서 찾음 $\rightarrow$ Skip 가능하거나 문맥으로 사용)
* Rank 3: **`Life Goes On`** (Doc A에 없었던 정보 $\rightarrow$ **Target**)

### 4.2 Expansion Query (확장 쿼리)
이전 단계에서 찾은 **Doc A(요약)**를 문맥으로 주입하여 2차 검색을 수행한다.

* **Formula:** $Q_2 = \text{Target} \oplus \texttt{[SEP]} \oplus E_1(\text{Summary})$
* **Result:** `Life Goes On [SEP] M.I.L.F.$ is a song by Fergie`

### 4.3 2nd Retrieval & Selection
* **Retrieved:** Doc C (`Life Goes On (song)`)
* **Selected Sentence ($E_2$):**
    * *"Life Goes On is a song by Fergie... released as the second single from Double Dutchess, following M.I.L.F.$."*

---

## ⚖️ Phase 5: Final Reasoning (최종 판결)

### 5.1 Evidence Integration (증거 통합)
1차 및 2차 검색에서 얻은 모든 증거를 결합한다.
$$E_{final} = E_1 \cup E_2$$

### 5.2 Final NLI Classification
* **Input:** `[CLS] Claim [SEP] E_final [SEP]`
    * *Claim:* "...produced by Polow da Don and followed by Life Goes On..."
    * *Evidence:* "...produced by Polow da Don..." ($E_1$) + "...following M.I.L.F.$..." ($E_2$)
* **Result:**
    * `Entailment`: **0.99**
    * `Neutral`: 0.00
    * `Contradiction`: 0.01

### 5.3 Verdict
**Label:** **SUPPORTS (참)**

---

## 🔁 6. Feedback & Backtracking Logic (피드백 및 역추적 요약)

시스템이 멈추지 않고 정답을 찾을 때까지 순환하는 **3가지 핵심 피드백 루프**를 시각화한다.

### 6.1 Logic Flowchart
```mermaid
graph TD
    Start([Start]) --> A[Phase 1: Priority Queue 생성]
    A --> B{Queue가 비었는가?}
    B -- Yes --> Fail([검증 실패 / NEI])
    B -- No --> C[Phase 2: 키워드 추출 & 1차 검색]
    
    C --> D{문서 신뢰도 > Threshold?}
    D -- No (문서 못 찾음) --> E[Backtrack: 다음 키워드 선정]
    E --> B
    
    D -- Yes (문서 찾음) --> F[Phase 3: Gatekeeper NLI 검증]
    
    F --> G{NLI Entailment Score?}
    
    G -- "High (> 0.9)" --> H([Phase 5: 최종 판결 (Stop)])
    
    G -- "Mid (0.3 ~ 0.9)" --> I[Phase 4: 문맥 확장 (Bridge 발견)]
    I --> J[이전 문서 요약 + 미해결 키워드로 2차 검색]
    J --> F
    
    G -- "Low (< 0.3)" --> K[Reject: 문서 폐기]
    K --> E