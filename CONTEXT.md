# TMDB RAG & Evaluation Harness (Maya)

An observable, multi-version RAG pipeline and evaluation harness for US theatrical releases (1970–2026), fronted by the "Maya" conversational agent with deterministic intent classification, multi-turn entity memory, and Langfuse-compatible telemetry, hosted on Hugging Face Spaces.

## Language

**Maya**:
The conversational film curator specialized in US TMDB movie releases from 1970 to 2026, executing structured intent routing, grounded response synthesis with movie poster rendering, and multi-turn state tracking.
_Avoid_: Chatbot, assistant, bot, LLM.

**Intent**:
The classified objective of a user utterance (one of `GREETING`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, `OUT_OF_SCOPE`) emitted by the Router.
_Avoid_: Action, task, query_type, category.

**Router**:
The temperature=0.0 LLM component that reads one user turn into a schema-validated structure: in v1 a routing decision (intent, filters, temporal constraints, coreference resolution) hardened by vocabulary gates; in v2 an Understanding, with Disposition and Turn Decision applied by code.
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

**Session Filter**:
The preferences Maya remembers for the whole session — moods, audiences, genres, directors, exclusions, year bounds — applied by code to every search until the visitor clears them. The chip labeled "Session filter" in the UI renders this set.
_Avoid_: Narrowing, persistent filter, memory filter.

**Current Filter**:
The effective filter set one search actually ran with — this turn's declared filters plus whatever Session Filter entries code folded in. The chip labeled "Current filter" in the UI renders this set; it is the ground truth of the WHERE clause, not the turn's raw reading.
_Avoid_: Turn filter, applied filter, query filter.

**Understanding**:
The single structured reading of one user turn against the Conversation State, returned by the Router in v2: Intent, Standalone Query, filters, era label, Preference Delta, reset signal, readiness to retrieve, missing Narrowing Axes, Clarifying Question, and Referenced Titles.
_Avoid_: Routing decision (the v1 object), extraction, parse, analysis.

**Preference Delta**:
What one user turn changed in the session's preferences: moods and audiences set or cleared, genres added or removed, exclusions added or revoked, caveats added. Merged into the Conversation State by code; never a replacement of the whole set.
_Avoid_: Preference update, new preferences, state patch.

**Mood Profile**:
A versioned, curated translation of one canonical mood into deterministic retrieval signals — phrasebook anchors for the semantic query, hard floors, and soft boosts — applied by code at the shared retrieve seam. Unknown moods resolve to none and change nothing.
_Avoid_: Mood embedding, vibe prompt, genre map (the hard genre filter is not a Mood Profile), text flavor.

**Mood Floor Criteria**:
Hard minimums a Mood Profile imposes on candidates (typically a vote-count floor). Tightening only — never loosens an explicit user filter; an empty pool after a floor stays empty.
_Avoid_: Quality gate, soft boost, fail-open floor.

**Narrowing Axis**:
One of the four things Maya may ask about before retrieving: mood, audience, genres, era. Directors and caveats can be stated but are never asked for.
_Avoid_: Slot, probe axis, funnel stage, dimension.

**Clarifying Question**:
Maya's one-sentence request for a missing Narrowing Axis, written by the Router in Maya's voice and shown only when the Turn Decision chooses to ask.
_Avoid_: Probe, prompt, follow-up, funnel question.

**Referenced Title**:
A title already shown in the session that the user points at ("the second one", "more like Holidate"), named by the Router so code can resolve it to a Movie Record.
_Avoid_: Focused movie (the resolved entity), mention, coreference.

**Disposition**:
The code step that enforces invariants on an Understanding before any decision: the 1970 dataset boundary, era label to years through Experiment Config, vocabulary membership, Preference Delta merge.
_Avoid_: Normalization, post-processing, validation, guard.

**Turn Decision**:
The code step that chooses ask, retrieve, converse, or pivot from a disposed Understanding and the narrowing knobs in Experiment Config.
_Avoid_: Routing, dispatch, edge, policy.

**Routing Stack**:
One complete, versioned way of turning a user turn into a retrieval or a reply: the Router prompt, its response model, the graph nodes and their wiring. Two stacks run side by side under Experiment Config while one is measured against the other; the loser is removed.
_Avoid_: Routing version, prompt version, pipeline, mode.

**Evaluation Harness**:
The benchmarking subsystem that executes standardized test sets across parameterized RAG pipeline versions to compute retrieval IR metrics (Hit Rate@K, MRR@K, Context Precision) and generation metrics (Faithfulness, Relevancy).
_Avoid_: Test runner, benchmark script, tester.

**Experiment Config**:
The dynamic schema of toggleable architectural knobs (models, reasoning effort, memory strategy, reformulation mode, embedder, hybrid alpha, reranker, guardrails) controllable via the Streamlit UI.
_Avoid_: App settings, parameters, flags.

**Trace**:
A complete user transaction log containing end-to-end latency, total token usage, estimated cost, intent classification, and child execution spans, mirrored to Langfuse and inspectable in the in-app DAG tree.
_Avoid_: Log entry, telemetry record.

**Feedback**:
A visitor's judgement on one Maya turn, always linked to that turn's Trace and recorded as one comment on the GitHub feedback inbox. Two kinds: a Rating and a Report.
_Avoid_: Review, vote, telemetry, re-route feedback (the Router's corrective prompt is not Feedback).

**Rating**:
A thumbs up or thumbs down on one Maya reply.
_Avoid_: Score, like, reaction.

**Report**:
Free text a visitor submits with `/feedback` about the last Maya reply, carrying its Feedback Window.
_Avoid_: Bug report, comment, note.

**Feedback Window**:
The last five user turns attached to a Report, each with its Intent, path, query and reply excerpts, and Trace id.
_Avoid_: Context, history dump.

**Feedback Action**:
What the loop did with a Feedback: received, promoted to an issue, or fixed.
_Avoid_: Status, outcome, resolution.

**Feedback Receipt**:
The persistent note under a reported Maya reply confirming the Report was recorded, linking its inbox comment when GitHub holds it.
_Avoid_: Toast, confirmation message, acknowledgement.

**FTS5 (Full-Text Search 5)**:
The native SQLite sparse lexical search engine executing BM25 keyword matching with Porter stemming over titles, overviews, directors, genres, and cast names to complement dense vector retrieval.
_Avoid_: Keyword searcher, regex search, text filter.

**FTS5 Shadow Tables**:
The internal SQLite storage tables (`movies_fts_data`, `movies_fts_idx`, `movies_fts_docsize`, `movies_fts_config`, `movies_fts_content`) automatically managed by SQLite to maintain inverted index postings, B-Tree lookups, and exact document length statistics for BM25 normalization.
_Avoid_: Extra tables, helper tables, secondary DBs.
