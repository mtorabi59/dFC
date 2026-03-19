# Install PydFC and Set Up Environment

Guide the user through installing PydFC.

## Context Sources

Use these when the user asks method or paper questions during setup:
- `docs/DFC_METHODS_CONTEXT.md`
- `docs/PAPER_KNOWLEDGE_BASE.md`

## Steps

1. Ask whether they use conda or pip.
2. Provide installation commands:

```bash
conda create --name pydfc_env python=3.11
conda activate pydfc_env
pip install pydfc
```

3. If they ask which method to start with, recommend SW as the default first run.
4. If they ask method assumptions/pros/cons, ground the answer in the context sources and avoid oversimplified claims.
5. If discussing paper-derived method assumptions, cite Torabi et al., 2024 when relevant.
