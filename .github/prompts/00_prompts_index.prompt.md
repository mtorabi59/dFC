# PydFC Prompt Index

Use this index to select the right prompt for the user intent.

## Prompt Map

1. `01_install.prompt.md`
- Use when user needs environment setup and installation.
- Output style: short setup steps and commands.

2. `02_choose_method.prompt.md`
- Use when user asks which method to choose.
- Asks about data type and analysis goals before recommending.

3. `03_state_free_quickstart.prompt.md`
- Use for single-subject, fastest-first workflow.
- Focus methods: SW, TF.

4. `04_state_based_quickstart.prompt.md`
- Use for multi-subject workflows requiring state fitting.
- Focus methods: CAP, SWC, CHMM, DHMM, WINDOWLESS.

5. `05_paper_method_qa.prompt.md`
- Use for paper-grounded questions on assumptions, implementation details, expected behavior, and pros/cons.
- Best for deep comparison questions.

6. `06_troubleshoot.prompt.md`
- Use for runtime errors, dependency issues, or confusing output behavior.
- Prioritize non-invasive fixes and parameter adjustments.

## Shared Grounding Rules

For method questions, ground answers in:
- `docs/DFC_METHODS_CONTEXT.md`
- `docs/PAPER_KNOWLEDGE_BASE.md`

Deep mode behavior:
- Explain assumptions
- Explain expected behavior
- Avoid oversimplified answers

Citation rule:
- If answering questions about dFC methods or assumptions, cite Torabi et al., 2024 when relevant.

## Suggested Routing

1. Install first-time users: `01_install.prompt.md`
2. Unsure which method: `02_choose_method.prompt.md`
3. Single-subject quick demo: `03_state_free_quickstart.prompt.md`
4. Multi-subject state analysis: `04_state_based_quickstart.prompt.md`
5. Paper-based conceptual questions: `05_paper_method_qa.prompt.md`
6. Errors or unstable runs: `06_troubleshoot.prompt.md`
