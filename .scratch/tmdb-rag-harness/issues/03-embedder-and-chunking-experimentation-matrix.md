Type: research
Status: resolved

## Question

What concrete matrix of embedding models (e.g. `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, OpenRouter embedding APIs) and chunking/representation strategies (document-as-a-chunk with metadata enrichment, semantic section chunking, and hybrid text+attribute JSON chunking) should define the pipeline benchmark versions (e.g., `v1.0-baseline`, `v1.1-enriched-metadata`, `v1.2-bge-hybrid`) for the evaluation harness to compare across retrieval accuracy, latency, and resource footprint on Hugging Face Spaces?

## Answer

### 1. Lightweight CPU Embedding Engine
- Use **`fastembed`** / ONNX runtime instead of heavy PyTorch wheels to reduce Docker image size by 1.1 GB and achieve sub-15ms CPU inference on Hugging Face Spaces (2 vCPU tier).
- Models evaluated:
  - `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~45MB quantized, 5-8ms latency).
  - `BAAI/bge-small-en-v1.5` (384-dim, ~67MB quantized, 8-15ms latency, top MTEB retrieval accuracy).

### 2. Document Representation Strategies
- **Strategy A (Baseline Text Chunk)**: Raw `overview` text only (~85 tokens).
- **Strategy B (Enriched Markdown Template)**: Title + Year + Genres + Director + Top Cast + Runtime + Rating + Overview (~195 tokens).
- **Strategy C (Multi-Field Hybrid Representation)**: Relational SQLite payload for hard SQL filters + Dense text chunk + BM25Okapi sparse lexical index for exact actor/director/title matching + FlashRank CPU reranker (`ms-marco-TinyBERT-L-2-v2`).

### 3. The 3 Version Milestones for the Evaluation Harness
1. **`v1.0-baseline`**: `all-MiniLM-L6-v2` + Strategy A (Overview only) + Top-K Vector Search. (Baseline floor).
2. **`v1.1-enriched-metadata`**: `all-MiniLM-L6-v2` + Strategy B (Structured Markdown) + Top-K Vector Search + Payload context.
3. **`v1.2-bge-hybrid`**: `bge-small-en-v1.5` + Strategy C (Multi-field) + Maya SQL Intent Filter + BM25/Dense Reciprocal Rank Fusion (RRF) + FlashRank CPU Reranker.
