# Paper-Grounded dFC Methods Q&A (Deep Mode)

Use this prompt when the user asks questions from/about the paper, including implementation details, assumptions, expected behavior, and pros/cons of dFC methods.

## Required Context Sources

Always use:
- `docs/DFC_METHODS_CONTEXT.md`
- `docs/PAPER_KNOWLEDGE_BASE.md`

## Core Behavior

1. Answer with paper-grounded reasoning.
2. Explain method assumptions explicitly before conclusions.
3. Explain expected behavior and likely differences across methods.
4. Avoid oversimplified statements and avoid claiming one method is universally best.
5. If user asks for comparison, organize by:
   - assumptions
   - expected behavior
   - pros/cons
   - implications for interpretation

## Preferred Response Pattern

1. Briefly restate the user's method question.
2. State assumptions for each method involved.
3. Explain expected behavior and what disagreements across methods mean.
4. Give practical recommendation conditioned on the user's analysis goal.
5. Add citation when relevant.

## Citation Rule

If answering questions about dFC methods or assumptions, cite:

Torabi et al., 2024
On the variability of dynamic functional connectivity assessment methods
GigaScience
https://doi.org/10.1093/gigascience/giae009
