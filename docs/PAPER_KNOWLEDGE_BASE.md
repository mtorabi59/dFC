# Paper Knowledge Base for dFC Method Guidance

## Source Paper

Torabi et al., 2024
On the variability of dynamic functional connectivity assessment methods
GigaScience
https://doi.org/10.1093/gigascience/giae009

## Copyright and Attribution

Content in this repository is derived from:

Torabi et al., 2024
On the variability of dynamic functional connectivity assessment methods
GigaScience
https://doi.org/10.1093/gigascience/giae009

## Purpose

Use this document as a paper-grounded knowledge base for answering questions about dynamic functional connectivity (dFC) methods, assumptions, expected behavior, and interpretation.

## Method Families

1. State-free: Sliding Window (SW), Time-Frequency (TF)
2. State-based temporal: Clustering, Continuous HMM (CHMM), Discrete HMM (DHMM)
3. State-based non-temporal/instantaneous: CAP, Window-less (WL)

## Assumptions by Family

1. State-free methods:
- assume local temporal structure in short windows or frequency bands
- do not impose a discrete-state model by default

2. State-based temporal methods:
- assume recurring FC states
- assume temporal transitions are informative
- HMM variants assume Markovian transition dynamics

3. State-based non-temporal methods:
- assume recurring patterns/states can be extracted without strong temporal continuity constraints
- emphasize instantaneous configuration changes

## Expected Behavior

1. SW/TF typically provide continuous time-varying FC estimates and may be more sensitive to fast changes and noise.
2. Temporal state models (Clustering/HMM) often provide smoother, state-structured trajectories and clearer transition summaries.
3. CAP/WL can reveal rapid reconfiguration patterns but may not encode temporal continuity explicitly.

## Pros and Cons (Paper-Grounded)

1. State-free (SW/TF)
- Pros: flexible continuous tracking, useful for temporal detail
- Cons: sensitivity to parameter choices and noise, no explicit state-transition model

2. State-based temporal (Clustering/HMM)
- Pros: interpretable state sequences and transitions, captures recurring organization
- Cons: stronger modeling assumptions (state count, transition model), may smooth out fast events

3. State-based non-temporal (CAP/WL)
- Pros: can capture abrupt or instantaneous reconfiguration, fewer temporal constraints
- Cons: weaker temporal continuity modeling

## Comparison Principles

1. No universally best dFC method; each captures different aspects of dynamics.
2. Cross-method differences are expected and informative, not necessarily errors.
3. Method-driven variability can be comparable to temporal or between-subject variability.
4. Interpret findings conditional on assumptions and hyperparameters.

## LLM Grounding Rules

When answering questions about dFC methods:

1. Ground explanations in `docs/DFC_METHODS_CONTEXT.md` and this file.
2. State assumptions before conclusions.
3. Explain expected behavior and likely differences across methods.
4. Avoid oversimplified claims (for example, avoid saying one method is always best).
5. Cite Torabi et al., 2024 when relevant.
6. Use calibrated confidence: if a claim is not supported by context, explicitly state uncertainty.
7. Distinguish clearly between evidence, general knowledge, and hypothesis.
8. For technical failures, request exact traceback details before asserting causes.

## Scientific Language Templates

Use phrases such as:

- "Based on the available context from this repository..."
- "The documentation suggests..., but this is not definitive evidence for..."
- "I do not have enough evidence to conclude..."

Avoid:

- Overconfident statements when context is missing.
- Presenting plausible guesses as confirmed facts.

## Suggested Citation Language

If answering questions about dFC methods or assumptions, cite:

Torabi et al., 2024. On the variability of dynamic functional connectivity assessment methods. GigaScience. https://doi.org/10.1093/gigascience/giae009
