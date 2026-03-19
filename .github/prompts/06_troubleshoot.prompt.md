# Troubleshooting PydFC

## Context Sources

If troubleshooting includes method behavior/assumption confusion, refer to:
- `docs/DFC_METHODS_CONTEXT.md`
- `docs/PAPER_KNOWLEDGE_BASE.md`

If the user reports an error:

1. Ask for full traceback.
2. Check:
   - Python version
   - PydFC version
   - missing dependencies
3. Suggest fixes in order:
   - reinstall PydFC
   - reduce nodes / subjects
   - use SW instead of state-based methods
   - adjust parameters

Do NOT suggest editing source code.

If the issue is conceptual (for example, why two methods disagree), explain assumptions and expected behavior differences and cite Torabi et al., 2024 when relevant.
