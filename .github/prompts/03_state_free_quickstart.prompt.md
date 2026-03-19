# State-Free Quickstart (Single Subject)

Guide the user through the fastest way to run PydFC.

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

1. Confirm PydFC is installed.
2. Provide demo data download commands.

### Jupyter
```python
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz -o sample_data/sub-0001_bold.nii.gz
```

3. Show minimal loading code for `BOLD` and ask whether they want SW or TF.
4. If the user asks SW vs TF, explain assumptions and expected behavior from the context sources.
5. Cite Torabi et al., 2024 when discussing paper-derived assumptions or tradeoffs.
