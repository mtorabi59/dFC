# State-Based Quickstart (Multi-Subject)

Guide the user through state-based analysis.

## Context Sources

Refer to:
- `docs/DFC_METHODS_CONTEXT.md` for assumptions, interpretation, and comparison principles
- `docs/PAPER_KNOWLEDGE_BASE.md` for paper-grounded implementation details and tradeoffs

Always ground method explanations in these documents.

## Deep Mode

When user asks about methods:
- Explain assumptions
- Explain expected behavior
- Avoid oversimplified answers

## Steps

1. Explain that state-based methods require multiple subjects.
2. Provide demo data download commands for 5 subjects.
3. Show minimal `BOLD_multi` loading example.
4. Ask which method they want:
   - CAP
   - SWC
   - CHMM
   - DHMM
   - WINDOWLESS
5. Offer a short description if they are unsure.
6. Provide one method snippet.
7. Ask if they want to try another method.

## Method Guidance Rules

1. Compare CAP, SWC, CHMM, DHMM, and WINDOWLESS by assumptions first, then expected behavior.
2. Do not claim one method is universally best.
3. If discussing paper-derived assumptions or pros/cons, cite Torabi et al., 2024 when relevant.
