Type: grilling
Status: resolved
Blocked by: 01, 02

## Question

What is the exact benchmark evaluation dataset (curating ground-truth query-answer-document pairs covering semantic, filter, superlative, and negation queries for 1970 US films, plus recent 2000–2026 out-of-scope verification queries), and what is the mathematical and programmatic definition of the evaluation metrics (Hit Rate@K, MRR@K, Context Precision, Faithfulness/Groundedness, Answer Relevancy, Intent Accuracy, Latency, Cost) to compute and serialize per version?

## Answer

### 1. Two-Tier Benchmark Evaluation Dataset (`data/eval_benchmark_dataset.json`)
The knowledge base remains strictly **1970 US TMDB releases**. To ensure test questions are easily verifiable by the user while testing both retrieval and guardrails, the benchmark is structured into two tiers:

#### Tier A: Recent & Familiar Movie Queries (2000–2026) — Guardrail & Out-of-Scope Suite (Majority)
- Well-known films (*Inception (2010)*, *The Dark Knight (2008)*, *Interstellar (2014)*, *Oppenheimer (2023)*, *Barbie (2023)*, *Everything Everywhere All at Once (2022)*, *Avatar (2009)*).
- **Purpose**: Evaluates Maya's **Intent Classification Accuracy**, **Temporal Mismatch Detection (`is_temporal_mismatch=True`)**, and **Zero-Hallucination Guardrail Rate** (verifying Maya never hallucinates answers for post-1970 films and correctly executes polite out-of-scope pivots).
- **Ground Truth**: `expected_intent: "OUT_OF_SCOPE"`, `requires_rag: false`, `ground_truth_movie_ids: []`.

#### Tier B: In-Scope 1970 Benchmark Suite — Retrieval & Synthesis Suite
- Curated 1970 queries covering Semantic Search (*Moon Zero Two*, *Marooned*), Attribute Filters (*Robert Altman*, *Jack Nicholson*), Superlatives (*Love Story*, *Airport*, *Waterloo*), and Negations.
- **Purpose**: Evaluates **Hit Rate@K**, **MRR@K**, **Context Precision**, and **Synthesis Faithfulness** against the 1970 knowledge base.

### 2. Formal Evaluation Metrics Suite
- **Retrieval Metrics**:
  - **Hit Rate@K**: $\frac{1}{|Q_{\text{in-scope}}|} \sum_{q} \mathbb{I}(\text{Top-}K_q \cap G_q \neq \emptyset)$
  - **MRR@K**: $\frac{1}{|Q_{\text{in-scope}}|} \sum_{q} \frac{1}{\text{rank}_q}$
  - **Context Precision@K**: Proportion of retrieved chunks relevant to query.
- **Guardrail & Generation Metrics**:
  - **Out-of-Scope Detection Precision & Recall**: Accuracy on 2000–2026 temporal queries.
  - **Faithfulness (Groundedness)**: LLM-as-a-Judge verifying zero hallucination against `<retrieved_movies>`.
  - **Intent Classification Accuracy**: Overall taxonomy accuracy across both tiers.
- **Operational Metrics**: Latency (ms), Token Usage, and OpenRouter Cost ($).

### 3. Metric Serialization & Decision Manifest
- Eval outputs saved to `evals/results/<version>.json`.
- `evals/decision_manifest.json` provides the historical version progression log.
- Vector database: Embedded **ChromaDB** (`chromadb.PersistentClient(path="./data/chroma_db")`) for versioned collections.
