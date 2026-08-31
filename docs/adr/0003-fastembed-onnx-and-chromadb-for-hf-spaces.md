# 3. FastEmbed ONNX & ChromaDB Local Collections for Hugging Face Spaces

We are using `fastembed` (Qdrant's lightweight ONNX runtime engine) with `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2` backed by local persistent **ChromaDB** collections (`data/chroma_db`) rather than heavy PyTorch packages or external cloud vector databases.

### Why this decision was made:
- Standard PyTorch wheel installations inflate Docker images by >1.2 GB and slow container build/boot times on Hugging Face Spaces. `fastembed` runs INT8 quantized ONNX models with zero PyTorch runtime dependencies.
- ONNX CPU inference executes in 5–15ms on 2 vCPU Hugging Face Space tiers.
- ChromaDB embedded mode (`chromadb.PersistentClient`) provides multi-collection versioning (`v1_0_baseline`, `v1_1_enriched`, `v1_2_bge_hybrid`) and metadata filtering with zero server hosting costs.
