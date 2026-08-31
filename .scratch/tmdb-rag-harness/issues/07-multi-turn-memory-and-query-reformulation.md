Type: grilling
Status: resolved

## Question

How should Maya handle multi-turn conversational memory (distinguishing short-term working context vs session state), history compression/truncation for long chats, coreference resolution (e.g. "who directed it?", "show me other movies with him"), fallback strategies when structured inference fails, and live experimentation knobs (e.g. 1-step fused vs 2-step dedicated rewriter) to ensure smooth, zero-latency multi-turn conversations on Hugging Face Spaces?

## Answer

### 1. The 5-Layer Conversational Memory Taxonomy (`ConversationState`)
Memory within `st.session_state` is partitioned into 5 explicit layers to prevent context bloat while maintaining entity continuity:
1. **Short-Term Working Window**: Last 4–6 raw user and assistant messages for immediate conversational rhythm.
2. **Active Entity Focus (`focused_entity` & `focused_person`)**: Tracks the primary film (`id`, `title`, `director`, `cast`) and person currently in discussion.
3. **Session-Level Exclusions & Preferences (`session_exclusions`)**: Persistent negative constraints (e.g. `excluded_genres: ["Horror"]`, `excluded_actors: ["Tom Cruise"]`) that automatically propagate to all future SQL `NOT LIKE` and vector filter operations.
4. **Seen Recommendation Tracker (`shown_movie_ids`)**: List of all TMDB IDs recommended during the session to eliminate redundant suggestions.
5. **Background Rolling Summary**: A 2-sentence summary generated if chat depth exceeds 10 turns to preserve thematic context without token overhead.

### 2. Multi-Turn Reformulation Engine (Configurable Knobs)
The architecture supports two toggleable reformulation modes in the Streamlit Experimentation Control Plane:
- **Mode A: Single Fused Router (Default / Low Cost / ~200ms)**: Passes the latest query + `ConversationState` to `meta-llama/llama-3.2-3b-instruct` to classify intent, extract filters, and generate the standalone search query in a single API call.
- **Mode B: Dedicated 2-Step Rewriter LLM (High Precision / ~450ms)**: Employs a specialized pre-processor LLM call solely dedicated to resolving complex conversational history into an expanded standalone search string before passing to the Router.

### 3. Graceful Multi-Turn Fallback Matrix
When structured query extraction encounters ambiguity, typos, or low confidence (< 0.70):
1. **Soft-Boosted Hybrid Fallback**: Converts hard SQL filters into scoring bonuses across BM25 and vector search so relevant candidates are always returned.
2. **Pure Dense Vector Fallback**: Embeds the raw conversational text against ChromaDB when JSON parsing fails completely.
3. **Interactive Clarification Trigger**: Prompts the user with a focused disambiguation question when multiple competing entities exist in prior context.

### 4. History Compression Policy
- Full fidelity retained for the last 6 messages.
- Messages older than 6 turns are truncated from the prompt, while the `ConversationState` preserves accumulated entity focus, active filters, and shown movie IDs.
