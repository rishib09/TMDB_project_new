Type: grilling
Status: resolved

## Question

How should we implement in-app user feedback (using Streamlit's native `st.feedback("thumbs")` per chat turn), associate each rating with its exact execution `trace_id` and RAG version, mirror user sentiment directly to Langfuse via `langfuse.score()`, store feedback in local SQLite for offline evaluation correlation, and render user satisfaction trends in the Evals Dashboard?

## Answer

### 1. In-App Feedback Mechanism (`st.feedback`)
- Render `st.feedback("thumbs", key=f"fb_{trace_id}")` below each Maya chat turn.
- Captures binary sentiment (+1 for thumbs up, 0 for thumbs down).

### 2. Dual Telemetry Sync & SQLite Storage
- **Langfuse Cloud**: Calls `langfuse_client.score(trace_id=trace_id, name="user_feedback", value=1.0 or 0.0)`.
- **Local SQLite Store**: Records to `user_feedback` table (`id`, `trace_id`, `rag_version`, `rating`, `timestamp`).

### 3. Evals Dashboard Analytics
- The **Evals Dashboard (Tab 2)** computes live "User Satisfaction %" aggregated by RAG version (`v1.0`, `v1.1`, `v1.2`).
