# PydFC Paper Figures

Scripts that generate publication-ready figures for the PydFC paper.

## Setup

Run scripts from the repository root:
```bash
python paper_figures/<script_name>.py
```

All outputs are saved to `paper_figures/output/`.

## Scripts

### `fig_method_showcase.py` — Figure A: Multi-Method dFC Showcase

Shows all 7 dFC methods applied to the same subject's resting-state fMRI data as a
grid of connectivity matrices (methods × time windows). Row labels are color-coded
by method family.

- **Data:** Downloads ~2 subjects from OpenNeuro AOMIC-PIOP1 automatically
- **Output:** `output/fig_method_showcase.png`
- **Runtime:** ~10–20 minutes (first run, includes download and computation)

### `fig_llm_tutorial_showcase.py` — Figure C: LLM Tutorial Showcase

Three-panel figure illustrating the AI-assisted tutorial layer:
1. Three access paths (Script / GitHub Copilot / Any LLM)
2. Example guided conversation (placeholder — replace before submission)
3. Example pydfc output (placeholder — replace before submission)

- **Data:** None required
- **Output:** `output/fig_llm_tutorial.png`
- **Runtime:** < 10 seconds

## Replacing Placeholders (Figure C)

Before submission, replace the two placeholder blocks in `fig_llm_tutorial_showcase.py`:

1. **Conversation (Panel 2):** Find the block marked
   `# --- PLACEHOLDER: REPLACE CONVERSATION BEFORE SUBMISSION ---`
   and update `CONVERSATION_LINES` with your actual LLM interaction.

2. **Output visualization (Panel 3):** Find the block marked
   `# --- PLACEHOLDER: REPLACE OUTPUT FIGURE BEFORE SUBMISSION ---`
   and replace `synthetic_matrix` with your actual dFC output data.
