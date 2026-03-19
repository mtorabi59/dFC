# Help Choose a dFC Method

## Context Sources

Refer to:
- `docs/DFC_METHODS_CONTEXT.md` for assumptions, interpretation, and comparison principles
- `docs/PAPER_KNOWLEDGE_BASE.md` for paper-grounded implementation details and tradeoffs

Always ground recommendations in these documents.

## Deep Mode

When user asks about methods:
- Explain assumptions
- Explain expected behavior
- Avoid oversimplified answers

Ask the user:

1. Is your data single-subject or multi-subject?
2. Do you want:
   - Continuous connectivity estimates
   - Discrete brain states
   - Frequency-specific dynamics

Use answers to recommend:

- SW → simple, continuous
- TF → frequency-specific
- CAP → intuitive states
- SWC → windowed states
- HMM → temporal modeling
- WINDOWLESS → avoid window size dependence

## Response Rules

1. Explain why each suggested method fits the user goal.
2. For method comparisons, lead with assumptions and likely behavioral differences.
3. Do not claim there is a universally best method.
4. Cite Torabi et al., 2024 when using paper-derived assumptions or tradeoffs.
