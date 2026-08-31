Type: grilling
Status: open

## Question

How should we select, prioritize, and structure the exact subset of TMDB metadata columns to embed for each embedding model's maximum token limit (256 tokens for `sentence-transformers/all-MiniLM-L6-v2` and 512 tokens for `BAAI/bge-small-en-v1.5`) to prevent tail truncation and avoid semantic dilution, while preserving purely tabular/numeric fields (budget, revenue, runtime, vote stats) in SQLite for deterministic SQL querying?
