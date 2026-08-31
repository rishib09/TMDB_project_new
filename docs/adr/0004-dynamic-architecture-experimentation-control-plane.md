# 4. Dynamic Architecture Experimentation Control Plane & Live Knobs

We are exposing core architectural choices (Router Model, Synthesis Model, Reasoning Effort, Reformulation Engine, Memory Strategy, Hybrid Alpha, Reranker ON/OFF, Guardrails) as a live `ExperimentConfig` schema accessible via the Streamlit Sidebar Lab and `/admin` chat shortcut.

### Why this decision was made:
- Rather than freezing a single pipeline implementation, the core value of this evaluation harness is educational: allowing users to toggle architectural knobs on the fly, observe real-time latency/cost changes in the Trace Inspector, and benchmark custom configurations against frozen baseline milestones.
- Toggling parameters alters the active execution DAG without requiring server restarts or code edits.
