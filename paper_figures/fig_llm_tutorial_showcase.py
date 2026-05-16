#!/usr/bin/env python3
"""
Figure C: LLM-Assisted Tutorial Showcase
=========================================
Three-panel figure illustrating the AI-assisted tutorial layer of PydFC:
  Panel 1 — Three access paths (Script / GitHub Copilot / Any LLM)
  Panel 2 — Example guided method-selection conversation (placeholder)
  Panel 3 — pydFC output from AI-generated code (placeholder)

No data required. Runs in <10 seconds.

Usage (from repo root):
    python paper_figures/fig_llm_tutorial_showcase.py

Output:
    paper_figures/output/fig_llm_tutorial.png

──────────────────────────────────────────────────────────────────────────────
REPLACING PLACEHOLDERS BEFORE SUBMISSION
──────────────────────────────────────────────────────────────────────────────
1. Conversation (Panel 2):
   Find the block   # --- PLACEHOLDER: REPLACE CONVERSATION BEFORE SUBMISSION ---
   and edit the CONVERSATION_LINES list with your actual LLM interaction.
   Each entry is a dict: {"role": "user"|"ai", "text": "...", "tags": [...]}
   Tags appear as small annotation chips below the message bubble.

2. Output figure (Panel 3):
   Find the block   # --- PLACEHOLDER: REPLACE OUTPUT FIGURE BEFORE SUBMISSION ---
   and replace the matrix or plot code with your actual pydfc visualization.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_GRAY = "#888888"
C_LIGHT = "#F5F5F5"
C_DARK = "#222222"

USER_BG = "#DCF8C6"  # light green — user bubble
AI_BG = "#FFFFFF"  # white — AI bubble
PANEL_BG = "#FAFAFA"

TAG_COLORS = {
    "⚠ assumption": "#E8A838",
    "✓ multi-method": "#55A868",
    "[ref] paper-grounded": "#4C72B0",
    "[safe] no code change": "#9B59B6",
}


# ════════════════════════════════════════════════════════════════════════════════
# Panel 1 — Three Access Paths
# ════════════════════════════════════════════════════════════════════════════════


def draw_panel1(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor(PANEL_BG)
    ax.patch.set_visible(True)

    # Panel title
    ax.text(
        0.5,
        0.97,
        "Three Access Paths",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=C_DARK,
        transform=ax.transAxes,
    )

    # ── Three boxes ──────────────────────────────────────────────────────────
    boxes = [
        {
            "x": 0.12,
            "y": 0.62,
            "label": "Script\nDemo",
            "sub": "examples/\ndFC_methods_demo.py",
            "tag": "Expert users",
            "color": C_BLUE,
            "entry": "Direct Python API",
        },
        {
            "x": 0.45,
            "y": 0.62,
            "label": "GitHub\nCopilot",
            "sub": ".github/prompts/\n02_choose_method.md",
            "tag": "Interactive learners",
            "color": C_ORANGE,
            "entry": "IDE chat interface",
        },
        {
            "x": 0.78,
            "y": 0.62,
            "label": "Any LLM\n+ SKILL.md",
            "sub": "docs/SKILL.md\n(copy-paste)",
            "tag": "Portable access",
            "color": C_GREEN,
            "entry": "ChatGPT / Claude / Gemini",
        },
    ]

    box_w = 0.22
    box_h = 0.22

    for b in boxes:
        xc, yc = b["x"], b["y"]

        # Main box
        rect = FancyBboxPatch(
            (xc - box_w / 2, yc - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.015",
            facecolor=b["color"],
            edgecolor="white",
            linewidth=1.5,
            transform=ax.transAxes,
            clip_on=False,
            alpha=0.92,
        )
        ax.add_patch(rect)

        # Label inside box
        ax.text(
            xc,
            yc + 0.012,
            b["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
        )

        # Sub-path below box label
        ax.text(
            xc,
            yc - box_h / 2 - 0.04,
            b["sub"],
            ha="center",
            va="top",
            fontsize=6.5,
            color=C_GRAY,
            style="italic",
            transform=ax.transAxes,
            linespacing=1.4,
        )

        # Tag chip above box
        ax.text(
            xc,
            yc + box_h / 2 + 0.04,
            b["tag"],
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=b["color"],
            transform=ax.transAxes,
        )

    # ── Arrows between boxes ──────────────────────────────────────────────────
    arrow_y = 0.62
    for x0, x1 in [(0.23, 0.34), (0.56, 0.67)]:
        ax.annotate(
            "",
            xy=(x1, arrow_y),
            xytext=(x0, arrow_y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.4, mutation_scale=12),
        )

    # ── Shared knowledge base ─────────────────────────────────────────────────
    kb_y = 0.22
    kb_rect = FancyBboxPatch(
        (0.05, kb_y - 0.07),
        0.90,
        0.14,
        boxstyle="round,pad=0.015",
        facecolor="#EEF3FB",
        edgecolor=C_BLUE,
        linewidth=1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(kb_rect)

    ax.text(
        0.5,
        kb_y + 0.02,
        "Shared Knowledge Base",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=C_BLUE,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        kb_y - 0.028,
        "DFC_METHODS_CONTEXT.md  ·  PAPER_KNOWLEDGE_BASE.md  ·  SKILL.md",
        ha="center",
        va="center",
        fontsize=7,
        color=C_GRAY,
        transform=ax.transAxes,
    )

    # Arrows from each box down to the knowledge base
    for xc in [0.12, 0.45, 0.78]:
        ax.annotate(
            "",
            xy=(xc, kb_y + 0.07),
            xytext=(xc, 0.62 - box_h / 2),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_GRAY,
                lw=1.0,
                mutation_scale=9,
                linestyle="dashed",
            ),
        )


# ════════════════════════════════════════════════════════════════════════════════
# Panel 2 — Example Conversation (Placeholder)
# ════════════════════════════════════════════════════════════════════════════════

# --- PLACEHOLDER: REPLACE CONVERSATION BEFORE SUBMISSION ---------------------
# Edit CONVERSATION_LINES to reflect your actual LLM interaction.
# Each entry:
#   "role"  : "user" or "ai"
#   "text"  : message text (use \n for line breaks within a bubble)
#   "tags"  : list of annotation tags drawn below the bubble (ai messages only)
#             Valid tag keys: "⚠ assumption", "✓ multi-method",
#                             "[ref] paper-grounded", "[safe] no code change"
CONVERSATION_LINES = [
    {
        "role": "user",
        "text": "I'm studying cognitive flexibility in\nhealthy adults using resting-state\nfMRI. Which dFC method should I use?",
        "tags": [],
    },
    {
        "role": "ai",
        "text": (
            "Good question. If you expect abrupt,\n"
            "discrete state changes, CAP or CHMM\n"
            "are good starting points.\n\n"
            "Note: CAP does not encode temporal\n"
            "ordering — if transition dynamics\n"
            "matter, prefer CHMM or SWC instead.\n\n"
            "A multi-method approach is recommended\n"
            "to distinguish biology from artifacts."
        ),
        "tags": ["⚠ assumption", "✓ multi-method", "[ref] paper-grounded"],
    },
    {
        "role": "user",
        "text": "Can you show me how to run CAP\nand CHMM with pydfc?",
        "tags": [],
    },
    {
        "role": "ai",
        "text": (
            "Sure. Here is a minimal example:\n\n"
            "  MEASURES_lst, _ = measures_initializer(\n"
            "    ['CAP', 'ContinuousHMM'],\n"
            "    params={'n_states': 12, ...},\n"
            "    alter_hparams={}\n"
            "  )\n"
            "  MEASURES_fit = estimate_group_FCS(\n"
            "    time_series=BOLD, MEASURES_lst=...\n"
            "  )\n\n"
            "Both methods use the same n_states,\n"
            "enabling controlled comparison."
        ),
        "tags": ["✓ multi-method", "[safe] no code change"],
    },
]
# --- END PLACEHOLDER ---------------------------------------------------------


def draw_chat_bubble(ax, y_frac, role, text, tags, bubble_width=0.78):
    """Draw a single chat bubble at vertical position y_frac (axes fraction)."""
    is_user = role == "user"
    bg_color = USER_BG if is_user else AI_BG
    border_color = "#B2DFA0" if is_user else "#DDDDDD"
    x_anchor = 0.96 if is_user else 0.04
    ha = "right" if is_user else "left"

    # Measure text height (approx: count lines)
    n_lines = text.count("\n") + 1
    bubble_h = 0.052 * n_lines + 0.04

    # Tag height (if any)
    tag_h = 0.038 if tags else 0.0

    total_h = bubble_h + tag_h + 0.015

    x0 = (x_anchor - bubble_width) if is_user else x_anchor
    y0 = y_frac - bubble_h

    # Background box
    rect = FancyBboxPatch(
        (x0, y0),
        bubble_width,
        bubble_h,
        boxstyle="round,pad=0.015",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=True,
    )
    ax.add_patch(rect)

    # Message text
    ax.text(
        x_anchor - 0.02 if is_user else x_anchor + 0.02,
        y_frac - bubble_h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=6.7,
        color=C_DARK,
        transform=ax.transAxes,
        fontfamily=(
            "monospace"
            if not is_user and "measures_initializer" in text
            else "sans-serif"
        ),
        linespacing=1.35,
    )

    # Tags
    tag_x = x0 + 0.01
    tag_y = y0 - 0.005
    for tag_key in tags:
        tag_color = TAG_COLORS.get(tag_key, C_GRAY)
        chip = FancyBboxPatch(
            (tag_x, tag_y - 0.028),
            len(tag_key) * 0.0085 + 0.02,
            0.025,
            boxstyle="round,pad=0.005",
            facecolor=tag_color,
            edgecolor="none",
            alpha=0.15,
            transform=ax.transAxes,
            clip_on=True,
        )
        ax.add_patch(chip)
        ax.text(
            tag_x + 0.01,
            tag_y - 0.015,
            tag_key,
            ha="left",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=tag_color,
            transform=ax.transAxes,
        )
        tag_x += len(tag_key) * 0.0085 + 0.035

    return total_h


def draw_panel2(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("#F0F2F5")
    ax.patch.set_visible(True)

    # Header bar
    header = FancyBboxPatch(
        (0, 0.93),
        1.0,
        0.07,
        boxstyle="square,pad=0",
        facecolor=C_BLUE,
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(header)
    ax.text(
        0.5,
        0.965,
        "PydFC AI Tutorial  ·  Method Selection",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
    )

    # Draw conversation bubbles top-to-bottom
    y = 0.90
    gap = 0.025  # vertical gap between bubbles
    for entry in CONVERSATION_LINES:
        text = entry["text"]
        n_lines = text.count("\n") + 1
        has_tags = bool(entry.get("tags"))
        bubble_h = 0.052 * n_lines + 0.04
        tag_h = 0.038 if has_tags else 0.0

        y -= gap
        draw_chat_bubble(ax, y, entry["role"], text, entry.get("tags", []))
        y -= bubble_h + tag_h + gap

    # Placeholder watermark
    ax.text(
        0.5,
        0.03,
        "[ PLACEHOLDER — replace with real conversation before submission ]",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#AAAAAA",
        style="italic",
        transform=ax.transAxes,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Panel 3 — Example pydFC Output (Placeholder)
# ════════════════════════════════════════════════════════════════════════════════

# --- PLACEHOLDER: REPLACE OUTPUT FIGURE BEFORE SUBMISSION --------------------
# Replace the synthetic matrix below with an actual dFC output.
# For example:
#   dFC_mat = dfc.get_dFC_mat(TRs=some_TRs)[0]   # (n_regions, n_regions)
# Then use ax.imshow(dFC_mat, cmap="seismic", ...) in draw_panel3().
np.random.seed(42)
_n = 20
_A = np.random.randn(_n, _n)
_cov = _A @ _A.T
_std = np.sqrt(np.diag(_cov))
PLACEHOLDER_MATRIX = _cov / np.outer(_std, _std)  # correlation matrix, shape (20, 20)
# --- END PLACEHOLDER ---------------------------------------------------------


def draw_panel3(ax):
    ax.set_facecolor(PANEL_BG)

    # Panel title
    ax.set_title(
        "pydFC Output\n(from AI-generated code)",
        fontsize=9,
        fontweight="bold",
        color=C_DARK,
        pad=6,
    )

    # ── dFC matrix ────────────────────────────────────────────────────────────
    inner_ax_pos = [
        0.08,
        0.30,
        0.84,
        0.62,
    ]  # [left, bottom, width, height] in axes coords
    inner_ax = ax.inset_axes(inner_ax_pos)

    v = float(np.percentile(np.abs(PLACEHOLDER_MATRIX), 97))
    im = inner_ax.imshow(
        PLACEHOLDER_MATRIX,
        cmap="seismic",
        vmin=-v,
        vmax=v,
        interpolation="nearest",
        aspect="equal",
    )
    inner_ax.set_xticks([])
    inner_ax.set_yticks([])
    inner_ax.set_xlabel("Brain regions", fontsize=7, labelpad=3)
    inner_ax.set_ylabel("Brain regions", fontsize=7, labelpad=3)
    inner_ax.set_title("dFC matrix at $t_k$", fontsize=8, pad=3, color=C_DARK)

    plt.colorbar(
        im, ax=inner_ax, shrink=0.75, pad=0.04, label="FC strength", location="right"
    )

    # ── Method info box ────────────────────────────────────────────────────────
    info_text = (
        "Method:  CAP  (n_states=12)\n" "Subject: sub-0001\n" "State:   FCS #3 of 12"
    )
    info_box = dict(
        boxstyle="round,pad=0.4", facecolor="#EEF3FB", edgecolor=C_BLUE, linewidth=0.8
    )
    ax.text(
        0.5,
        0.16,
        info_text,
        ha="center",
        va="center",
        fontsize=7,
        color=C_DARK,
        bbox=info_box,
        transform=ax.transAxes,
        fontfamily="monospace",
        linespacing=1.5,
    )

    # ── Placeholder watermark ──────────────────────────────────────────────────
    ax.text(
        0.5,
        0.04,
        "[ PLACEHOLDER — replace with actual dFC output ]",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#AAAAAA",
        style="italic",
        transform=ax.transAxes,
    )

    ax.set_axis_off()


# ════════════════════════════════════════════════════════════════════════════════
# Assemble figure
# ════════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Figure C: LLM Tutorial Showcase")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 7.5), facecolor="white")

    # Three panels with different widths: 28% | 40% | 32%
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.8, 4.0, 3.2],
        left=0.01,
        right=0.99,
        bottom=0.04,
        top=0.93,
        wspace=0.04,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Draw panels
    draw_panel1(ax1)
    draw_panel2(ax2)
    draw_panel3(ax3)

    # Panel labels (a), (b), (c)
    for ax, letter in zip([ax1, ax2, ax3], ["(a)", "(b)", "(c)"]):
        ax.text(
            -0.02,
            1.01,
            letter,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=C_DARK,
            transform=ax.transAxes,
        )

    # Figure title
    fig.text(
        0.5,
        0.975,
        "AI-Assisted Tutorial Layer in PydFC",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=C_DARK,
    )

    # Thin borders around panels
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#CCCCCC")

    output_path = os.path.join(OUTPUT_DIR, "fig_llm_tutorial.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved → {output_path}")
    print("\nNote: Panels (b) and (c) contain placeholders.")
    print("See README.md for instructions on replacing them before submission.")


if __name__ == "__main__":
    main()
