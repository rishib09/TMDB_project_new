Type: research
Status: resolved

## Question

Is it feasible to lean out `ConversationState.add_turn()` using LangGraph's native state reducers (`add_messages`, custom reducers for `shown_movie_ids`, `session_preferences`, and `session_tokens`)?

## Answer

### Feasibility: 100% Feasible & Recommended
Instead of an imperative 60-line `.add_turn()` method, we can define `MayaGraphState` using LangGraph 0.2+ functional reducers:
- `messages: Annotated[Sequence[BaseMessage], add_messages]`
- `shown_movie_ids: Annotated[List[int], merge_unique_ids]`
- `session_preferences: Annotated[UserSessionPreferences, merge_preferences]`
- `session_tokens: Annotated[int, operator.add]`

### Benefits:
1. **Zero State Mutation Boilerplate**: LangGraph nodes simply return dictionaries (e.g. `{"messages": [AIMessage(...)], "shown_movie_ids": [m.id]}`), and reducers handle deduplication, merging, and appending.
2. **Streamlit Leanness**: Streamlit only manages `st.session_state.thread_id`, and `MemorySaver` handles multi-turn state persistence.
3. **Langfuse Cloud Tracing**: Native compatibility with Langfuse Chat Inspector view.
