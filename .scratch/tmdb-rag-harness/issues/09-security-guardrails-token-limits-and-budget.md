Type: grilling
Status: resolved

## Question

How should we architect the security and guardrail layer to gracefully mitigate prompt injection/jailbreaks, detect and deflect off-topic/adversarial queries with film-curator persona pivots, enforce a hard session token cap (e.g. 15,000 tokens/session), and track a persistent weekly API expenditure ceiling (e.g. $5.00/week) stored in SQLite with proactive graceful throttling?

## Answer

### 1. Prompt Injection & Off-Topic Deflection
- **Sanitizer & Router Filter**: Fast regex strips delimiter override attempts (`</retrieved_movies>`, `system:` overrides). Adversarial and off-topic queries are classified as `OUT_OF_SCOPE` (`requires_rag=false`).
- **Graceful Film-Themed Pivot**: Maya responds with polite in-character deflection:
  > *"I am Maya, your cinema guide for US films (1970–2026)! While I can't assist with general programming or off-topic queries, if you'd like to explore movies about technology, sports, or sci-fi, I'm here to help!"*

### 2. Session Token Cap & Persistent Weekly Budget Ceiling
- **Session Token Cap (15,000 tokens)**: `st.session_state["session_tokens"]` tracks accumulated tokens. If exceeded, prompts user to click "🔄 Reset Session".
- **Weekly Budget Ceiling ($5.00/week)**: SQLite table `budget_tracker` logs daily costs. If rolling 7-day total reaches $5.00, app prompts user to enter their personal OpenRouter API key in the sidebar.
