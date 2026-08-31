# TMDB RAG & Evaluation Harness (Maya)

An observable, multi-version RAG pipeline and evaluation harness for US theatrical releases (1970–2026), fronted by the "Maya" conversational agent with deterministic intent classification, multi-turn entity memory, and Langfuse-compatible telemetry, hosted on Hugging Face Spaces.

## Language

**Maya**:
The conversational film curator specialized in US TMDB movie releases from 1970 to 2026, executing deterministic intent routing, grounded response synthesis with movie poster rendering, and multi-turn state tracking.
_Avoid_: Chatbot, assistant, bot, LLM.

**Intent**:
The classified objective of a user utterance (one of `GREETING`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, `OUT_OF_SCOPE`) emitted by the deterministic router.
_Avoid_: Action, task, query_type, category.

**Router**:
The temperature=0.0 LLM component executing structured Pydantic schema validation to classify user intent, extract discrete SQL filters, detect temporal constraints, and resolve conversational coreferences in a single pass.
_Avoid_: Classifier, prompt classifier, dispatcher.

**Standalone Query**:
The self-contained, disambiguated search string produced during query reformulation with conversational pronouns and ellipses resolved.
_Avoid_: Rewritten query, cleaned prompt.

**Movie Record**:
The normalized domain entity representing a film (`id`, `title`, `release_date`, `release_year`, `genres`, `director`, `cast`, `runtime`, `budget`, `revenue`, `vote_average`, `vote_count`, `overview`, `keywords`, `poster_path`, `poster_url`).
_Avoid_: Film object, movie item, document.

**Conversation State**:
The multi-turn session data structure tracking the sliding message window, active focused movie entity, focused person, persistent user exclusions, and recommended movie history.
_Avoid_: Chat history, session memory, context buffer.

**Persistent Exclusion**:
A user-specified negative preference (e.g. `excluded_genres: ["Horror"]`, `excluded_actors: ["Tom Cruise"]`) that remains active across all subsequent retrieval queries within a session until explicitly revoked.
_Avoid_: Permanent filter, blacklist, negative prompt.

**Evaluation Harness**:
The benchmarking subsystem that executes standardized test sets across parameterized RAG pipeline versions to compute retrieval IR metrics (Hit Rate@K, MRR@K, Context Precision) and generation metrics (Faithfulness, Relevancy).
_Avoid_: Test runner, benchmark script, tester.

**Experiment Config**:
The dynamic schema of toggleable architectural knobs (models, reasoning effort, memory strategy, reformulation mode, embedder, hybrid alpha, reranker, guardrails) controllable via the Streamlit UI.
_Avoid_: App settings, parameters, flags.

**Trace**:
A complete user transaction log containing end-to-end latency, total token usage, estimated cost, intent classification, and child execution spans, mirrored to Langfuse and inspectable in the in-app DAG tree.
_Avoid_: Log entry, telemetry record.

**FTS5 (Full-Text Search 5)**:
The native SQLite sparse lexical search engine executing BM25 keyword matching with Porter stemming over titles, overviews, directors, genres, and cast names to complement dense vector retrieval.
_Avoid_: Keyword searcher, regex search, text filter.

**FTS5 Shadow Tables**:
The internal SQLite storage tables (`movies_fts_data`, `movies_fts_idx`, `movies_fts_docsize`, `movies_fts_config`, `movies_fts_content`) automatically managed by SQLite to maintain inverted index postings, B-Tree lookups, and exact document length statistics for BM25 normalization.
_Avoid_: Extra tables, helper tables, secondary DBs.
