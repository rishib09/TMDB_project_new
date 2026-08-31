Type: research
Status: resolved

## Question

Is it feasible to add a dynamic `token_budget` parameter (256, 512, 1024, 2048, 4096 tokens) to `ExperimentConfig`, and how should `MovieRecord.to_dense_text(strategy, token_budget)` dynamically adapt its metadata payload (cast depth, keywords, production notes) to match the context limits of different embedding models (MiniLM 256 vs BGE-Small 512 vs modern 8k embedders)?

## Answer

### Feasibility: 100% Feasible & Recommended
1. **Model Context Ceilings**:
   - `all-MiniLM-L6-v2`: Hard limit **256 tokens** (setting >256 causes silent truncation).
   - `BAAI/bge-small-en-v1.5`: Hard limit **512 tokens**.
   - `nomic-ai/nomic-embed-text-v1.5`: Up to **8192 tokens**.
2. **Prioritized Tier-Packing Algorithm in `MovieRecord.to_dense_text()`**:
   - **256 tokens**: Core identity (Title, Year, Director, Genres) + top 4 cast + top 6 keywords + trimmed synopsis (~120 words).
   - **512 tokens**: Core identity + top 10 cast with character roles + all keywords + tagline + full unabridged synopsis + runtime/ratings.
   - **1024+ tokens**: Full cast (top 20) + crew + production studios + financials (budget/revenue) + full synopsis.
3. **ExperimentConfig**: Add `token_budget: int = Field(default=256, ...)` with automatic ceiling validation against the active embedding model.
