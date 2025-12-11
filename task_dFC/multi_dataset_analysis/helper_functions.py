import colorsys
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors

###################### Publication style ######################


def setup_pub_style():
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update(
        {
            # Fonts & text
            "font.size": 18,  # base
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            # Lines/markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            # Figure/layout
            "figure.dpi": 150,  # on-screen
            "savefig.dpi": 500,  # export
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            # Vector export: keep text as text in PDF/SVG
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig_pub(path_png_or_pdf: str):
    Path(Path(path_png_or_pdf).parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_png_or_pdf)
    # # Also export vector PDF alongside PNG unless you passed a .pdf
    # p = Path(path_png_or_pdf)
    # if p.suffix.lower() != ".pdf":
    #     plt.savefig(p.with_suffix(".pdf"))


###################### ml_results ######################


def get_cog_domain_info(simul_or_real: str):
    """
    Return:
        DOMAIN_ORDER: list of domains in preferred order
        TASK2DOMAIN: dict mapping canonical task codes to domains
        DOMAIN_BASE: dict mapping domains to base colors (hex)
    """
    if simul_or_real == "real":
        # --- Cognitive-Atlas–aligned domains (order on paper) ---
        DOMAIN_ORDER = [
            "Arousal & Regulatory Systems",
            "Cognitive Systems",
            "Negative Valence System",
            "Positive Valence System",
            "Sensorimotor Systems",
        ]

        # --- Map canonical task codes -> domain ---
        TASK2DOMAIN = {
            # Language & Regulatory Systems
            "emotionregulation": "Arousal & Regulatory Systems",
            # Cognitive Systems
            "audsem": "Cognitive Systems",
            "visrhyme": "Cognitive Systems",
            "vissem": "Cognitive Systems",
            "visspell": "Cognitive Systems",
            "arithmetic": "Cognitive Systems",
            "stroop": "Cognitive Systems",
            "cuedts": "Cognitive Systems",
            "axcpt": "Cognitive Systems",
            "matching": "Cognitive Systems",
            "stern": "Cognitive Systems",
            "st": "Cognitive Systems",
            "vswm": "Cognitive Systems",
            "expo": "Cognitive Systems",
            "recall": "Cognitive Systems",
            "feedback": "Cognitive Systems",
            "ppalocalizer": "Cognitive Systems",
            "localiser": "Cognitive Systems",
            "localizer": "Cognitive Systems",
            # Positive Valence System
            "cic": "Positive Valence System",
            "fribbids": "Positive Valence System",
            "risk": "Positive Valence System",
            "itc": "Positive Valence System",
            # Negative Valence System
            "fearlearning": "Negative Valence System",
            "paingen": "Negative Valence System",
            # Sensorimotor
            "motor": "Sensorimotor Systems",
            "execution": "Sensorimotor Systems",
            "imagery": "Sensorimotor Systems",
            "ihg": "Sensorimotor Systems",
        }
        # base colors per domain (distinct, colorblind-friendly)
        DOMAIN_BASE = {
            "Arousal & Regulatory Systems": "#9467bd",
            "Cognitive Systems": "#ff7f0e",
            "Positive Valence System": "#02833E",
            "Negative Valence System": "#d62728",
            "Sensorimotor Systems": "#1f77b4",
        }
    elif simul_or_real == "simulated":
        # --- Categories of simulated task paradigms ---
        DOMAIN_ORDER = [
            "Simulated Periodic",
            "Strong Performance on Real Data",
            "Weak Performance on Real Data",
        ]
        # --- Map task codes -> category ---
        TASK2DOMAIN = {
            # Simulated Periodic
            "lowfreqlongrest": "Simulated Periodic",
            "lowfreqshortrest": "Simulated Periodic",
            "lowfreqshorttask": "Simulated Periodic",
            # Optimal Paradigm Design, Strong Performance on Real Data
            "axcpt": "Strong Performance on Real Data",
            "stern": "Strong Performance on Real Data",
            "cuedts": "Strong Performance on Real Data",
            # Optimal Paradigm Design, Weak Performance on Real Data
            "execution": "Weak Performance on Real Data",
            "imagery": "Weak Performance on Real Data",
            "localizer": "Weak Performance on Real Data",
            "ppalocalizer": "Weak Performance on Real Data",
            # Sub-Optimal Paradigm Design, Weak Performance on Real Data
            "itc": "Weak Performance on Real Data",
            "stroop": "Weak Performance on Real Data",
            "risk": "Weak Performance on Real Data",
        }
        # base colors per domain (distinct, colorblind-friendly)
        DOMAIN_BASE = {
            "Simulated Periodic": "#1f77b4",
            "Strong Performance on Real Data": "#02833E",
            "Weak Performance on Real Data": "#d62728",
        }
    else:
        raise ValueError(f"Invalid simul_or_real: {simul_or_real}")
    return DOMAIN_ORDER, TASK2DOMAIN, DOMAIN_BASE


def canon_task(task_str: str) -> str:
    """strip 'task-' and non-letters, lowercase → canonical key"""
    s = task_str.replace("task-", "")
    s = re.sub(r"[^a-zA-Z]", "", s)
    return s.lower()


def task_domain_real(task: str) -> str:
    _, TASK2DOMAIN, _ = get_cog_domain_info("real")
    return TASK2DOMAIN.get(canon_task(task), "Other")


def task_domain_simul(task: str) -> str:
    _, TASK2DOMAIN, _ = get_cog_domain_info("simulated")
    return TASK2DOMAIN.get(canon_task(task), "Other")


def shade_series_same_hue(base_hex: str, n: int, delta_L=0.08, delta_S=0.06):
    """
    Same hue; small, symmetric tweaks in lightness/saturation → very similar colors.
    delta_L/S control how similar the shades are (smaller = more similar).
    """
    if n <= 1:
        return [base_hex]
    r, g, b = mcolors.to_rgb(base_hex)
    # colorsys uses HLS (Hue, Lightness, Saturation)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # symmetric lightness offsets around original l
    offs_L = np.linspace(-delta_L, +delta_L, n)
    # small saturation jitter to avoid identical look
    offs_S = np.linspace(-delta_S, +delta_S, n)

    cols = []
    for dL, dS in zip(offs_L, offs_S):
        li = float(np.clip(l + dL, 0.05, 0.95))
        si = float(np.clip(s + dS, 0.20, 0.95))
        r2, g2, b2 = colorsys.hls_to_rgb(h, li, si)
        cols.append(mcolors.to_hex((r2, g2, b2)))
    return cols


def build_task_order_and_palette(
    tasks_iterable, simul_or_real, similarity_L=0.08, similarity_S=0.06
):
    """Domain-first task order + very-similar shades per domain."""
    tasks = list(tasks_iterable)
    if simul_or_real == "real":
        dom_of = {t: task_domain_real(t) for t in tasks}
    elif simul_or_real == "simulated":
        dom_of = {t: task_domain_simul(t) for t in tasks}

    DOMAIN_ORDER, _, DOMAIN_BASE = get_cog_domain_info(simul_or_real)
    # order: by DOMAIN_ORDER, then alphabetical within domain
    task_order = []
    for dom in DOMAIN_ORDER:
        ts = sorted([t for t in tasks if dom_of[t] == dom], key=lambda s: s.lower())
        task_order.extend(ts)

    # palette: near-identical shades per domain
    palette = {}
    for dom in DOMAIN_ORDER:
        ts = [t for t in task_order if dom_of.get(t, "Other") == dom]
        if not ts:
            continue
        shades = shade_series_same_hue(
            DOMAIN_BASE[dom], len(ts), delta_L=similarity_L, delta_S=similarity_S
        )
        for t, col in zip(ts, shades):
            palette[t] = col
    return task_order, palette


def domain_sorted_rows(index_tasks, TASKS_to_include, simul_or_real):
    # preserve only tasks present in the matrix
    present = [t for t in index_tasks if t in TASKS_to_include]
    # if simul_or_real != "real":
    #     return sorted(present, key=lambda s: s.lower())
    # domain-first, then alphabetical
    if simul_or_real == "real":
        dom_of = {t: task_domain_real(t) for t in present}
    elif simul_or_real == "simulated":
        dom_of = {t: task_domain_simul(t) for t in present}
    DOMAIN_ORDER, _, _ = get_cog_domain_info(simul_or_real)
    ordered = []
    for dom in DOMAIN_ORDER:
        ts = sorted([t for t in present if dom_of[t] == dom], key=lambda s: s.lower())
        ordered.extend(ts)
    return ordered


def boldify_axes(ax, xlabel=None, ylabel=None, rotate_xticks=35):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")
    # dFC method names on x-axis
    if rotate_xticks is not None:
        plt.setp(
            ax.get_xticklabels(), fontweight="bold", rotation=rotate_xticks, ha="right"
        )
    else:
        plt.setp(ax.get_xticklabels(), fontweight="bold")


def draw_grouped_legend_panel(
    ax_leg,
    task_order,
    domain_of,
    palette,
    domain_order,
    ncols=2,
    fontsize=8,
    markersize=5,
    colpad=0.04,
):
    ax_leg.set_axis_off()
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    items = []
    for dom in domain_order:
        ts = [t for t in task_order if domain_of.get(t, "Other") == dom]
        if not ts:
            continue
        items.append(("header", dom))
        items.extend(("task", t) for t in ts)

    rows = len(items)
    rows_per_col = max(1, math.ceil(rows / ncols))
    x_cols = [0.02 + i * (1.0 / ncols) for i in range(ncols)]
    top = 0.98
    dy = (top - 0.06) / rows_per_col

    col = 0
    row_in_col = 0
    for kind, val in items:
        if row_in_col >= rows_per_col:
            col += 1
            row_in_col = 0
        if col >= ncols:
            break
        x = x_cols[col]
        y = top - row_in_col * dy
        if kind == "header":
            ax_leg.text(
                x, y, val, fontsize=fontsize, fontweight="bold", ha="left", va="top"
            )
        else:
            t = val
            color = palette.get(t, "0.4")
            ax_leg.plot(
                [x],
                [y],
                marker="o",
                ms=markersize,
                mfc=color,
                mec="#222222",
                mew=0.8,
                ls="None",
            )
            ax_leg.text(x + colpad, y, t, fontsize=fontsize, ha="left", va="center")
        row_in_col += 1


def mean_ci_boot(y, n_boot=3000, ci=95, rng=None):
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(y))
    if y.size == 1:
        return m, m, m
    if rng is None:
        rng = np.random.default_rng()  # fresh entropy
    idx = rng.integers(0, y.size, size=(n_boot, y.size))
    boots = np.mean(y[idx], axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return m, lo, hi


def summarize_methods_across_tasks(
    df_plot, ycol, method_col="dFC method", ci_func=mean_ci_boot
):
    """
    Return a DataFrame with columns: [method_col, 'mean','lo','hi'].
    Robust to Pandas quirks; no MultiIndex/unnamed columns.
    Assumes df_plot has one row per (task, method) already (your BEST table).
    """
    rows = []
    for meth, s in df_plot.groupby(method_col, observed=True)[ycol]:
        m, lo, hi = ci_func(s.values)
        rows.append({method_col: meth, "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def overlay_method_mean_ci(
    ax,
    df_plot,
    ycol,
    method_col="dFC method",
    line_halfwidth=0.30,
    cap_halfwidth=0.12,
    color="#222",
    lower=None,
    upper=None,
    rng=None,
):
    # map x positions from current ticks (call after you set/rotate xticklabels)
    xticks = ax.get_xticks()
    xlabs = [t.get_text() for t in ax.get_xticklabels()]
    xpos = {lab: xticks[i] for i, lab in enumerate(xlabs)}

    # summarize robustly
    summ = summarize_methods_across_tasks(
        df_plot, ycol, method_col, ci_func=lambda y: mean_ci_boot(y, rng=rng)
    )

    # clip to metric bounds if provided
    def clip(v):
        if lower is not None:
            v = max(lower, v)
        if upper is not None:
            v = min(upper, v)
        return v

    for _, r in summ.iterrows():
        meth = r[method_col]
        if meth not in xpos or np.isnan(r["mean"]):
            continue
        x = xpos[meth]
        m = clip(r["mean"])
        lo = clip(r["lo"]) if not np.isnan(r["lo"]) else m
        hi = clip(r["hi"]) if not np.isnan(r["hi"]) else m

        # mean line (thick) + CI whisker & caps (thin)
        ax.hlines(
            m, x - line_halfwidth, x + line_halfwidth, colors=color, lw=2.6, zorder=6
        )
        ax.vlines(x, lo, hi, colors=color, lw=1.2, alpha=0.9, zorder=5)
        ax.hlines(
            [lo, hi],
            x - cap_halfwidth,
            x + cap_halfwidth,
            colors=color,
            lw=1.2,
            alpha=0.9,
            zorder=5,
        )


def wrap_domain(dom: str, max_len: int = 20) -> str:
    # First, break on the preferred delimiters
    s = dom.replace(" & ", " &\n").replace(", ", ",\n")
    out = []
    for seg in s.splitlines():
        # Then wrap remaining long segments on spaces (no hard splits)
        wrapped = textwrap.wrap(
            seg, width=max_len, break_long_words=False, break_on_hyphens=True
        )
        out.extend(wrapped if wrapped else [""])
    return "\n".join(out)


def add_domains_between_ylabel_and_ticks(
    ax,
    row_order,
    task_to_domain,
    label_rotation=30,
    tick_pad_pts=28,
    ylabel_pad_pts=60,
    domain_x_frac=-0.11,  # x position for the domain column (axes frac)
    left_extend_frac=0.02,  # how far past the text the line extends
    label_x_offset_frac=0.008,  # small nudge right from domain_x_frac
    label_align="left",  # "left" | "center" | "right"
    label_kw=None,
    sep_kw=None,
):
    if label_kw is None:
        label_kw = dict(
            fontsize=10, fontweight="bold", color="#222", ha="left", va="center"
        )  # default to left
    else:
        # override HA with requested alignment but keep user's other styles
        label_kw = {
            **label_kw,
            "ha": {"left": "left", "center": "center", "right": "right"}[label_align],
            "va": "center",
        }
    if sep_kw is None:
        sep_kw = dict(color="#777", lw=1.0, alpha=0.9)

    if not row_order:
        return

    ax.tick_params(axis="y", pad=tick_pad_pts)
    ax.yaxis.labelpad = ylabel_pad_pts

    # row centers (as before) ...
    yticks = ax.get_yticks()
    yticklabs = [t.get_text() for t in ax.get_yticklabels()]
    if yticklabs and len(yticklabs) == len(row_order):
        lbl2y = {lab: y for lab, y in zip(yticklabs, yticks)}
        y_centers = [lbl2y.get(t, np.nan) for t in row_order]
    else:
        n = len(row_order)
        y0, y1 = ax.get_ylim()
        base = np.linspace(0.5, n - 0.5, n)
        if y1 < y0:
            scale = (y0 - y1) / (n - 1)
            y_centers = y0 - (base - 0.5) * scale
        else:
            scale = (y1 - y0) / (n - 1)
            y_centers = y0 + (base - 0.5) * scale

    doms = [task_to_domain.get(t, "Other") for t in row_order]
    blocks = []
    start = 0
    for i in range(1, len(doms)):
        if doms[i] != doms[i - 1]:
            blocks.append((doms[start], start, i - 1))
            start = i
    if len(doms):
        blocks.append((doms[start], start, len(doms) - 1))

    trans_text = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for dom, i0, i1 in blocks:
        y_block = float(np.nanmean(y_centers[i0 : i1 + 1]))
        # left-aligned text slightly to the right of the domain column anchor
        x_text = domain_x_frac + (label_x_offset_frac if label_align == "left" else 0.0)
        dom_updated = wrap_domain(dom, max_len=24)
        ax.text(
            x_text,
            y_block,
            dom_updated,
            rotation=label_rotation,
            transform=trans_text,
            clip_on=False,
            **label_kw,
        )

    # separators (heatmap + extension into the domain column)
    x_min, x_max = ax.get_xlim()
    trans_sep = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i in range(len(doms) - 1):
        if doms[i + 1] != doms[i]:
            y_sep = 0.5 * (y_centers[i] + y_centers[i + 1])
            ax.hlines(y_sep, x_min, x_max, **sep_kw)  # inside heatmap
            ax.plot(
                [0.0, domain_x_frac - left_extend_frac],
                [y_sep, y_sep],
                transform=trans_sep,
                clip_on=False,
                **sep_kw,
            )  # into domain column


###################### task_timing_stats ######################


def as_long_df(d, value_col, task_col="task"):
    rows = []
    for t, vals in d.items():
        for v in vals:
            rows.append({task_col: t, value_col: v})
    return pd.DataFrame(rows)


# --- median labels with matching hue colors (log-safe) ---
def annotate_medians_by_geometry(
    ax,
    df_long,
    x_col,
    hue_col,
    y_col,
    x_order,
    hue_order,
    fmt="{:.0f}",  # ints; change to "{:.2g}" if you prefer
    y_nudge_factor=1.08,
    bin_halfwidth=0.6,
    bbox_alpha=0.9,
):
    def _luminance(r, g, b):
        # simple relative luminance for contrast
        return 0.299 * r + 0.587 * g + 0.114 * b

    # collect box patches and centers
    patches = [
        p for p in getattr(ax, "artists", []) if isinstance(p, mpl.patches.PathPatch)
    ]
    if not patches:
        patches = [p for p in ax.patches if isinstance(p, mpl.patches.PathPatch)]

    boxes = []
    for p in patches:
        verts = p.get_path().vertices
        xs = verts[:, 0]
        x_center = 0.5 * (xs.min() + xs.max())
        boxes.append((x_center, p))

    if not boxes:
        return

    # bin by x tick index (0..len(x_order)-1)
    boxes_by_tick = {i: [] for i in range(len(x_order))}
    for x_center, p in boxes:
        idx = int(round(x_center))
        if idx in boxes_by_tick and abs(x_center - idx) <= bin_halfwidth:
            boxes_by_tick[idx].append((x_center, p))

    # medians from data
    med_dict = df_long.groupby([x_col, hue_col])[y_col].median().to_dict()

    for i, task in enumerate(x_order):
        group = boxes_by_tick.get(i, [])
        if not group:
            continue
        # left->right inside this task bin
        group.sort(key=lambda t: t[0])

        for j, hue in enumerate(hue_order):
            if j >= len(group):
                break
            x_center, patch = group[j]
            med = med_dict.get((task, hue), np.nan)
            if not (np.isfinite(med) and med > 0):
                continue

            # extract the exact facecolor of this box (matches legend/palette)
            fc = patch.get_facecolor()  # RGBA
            if fc is None or len(fc) < 3:
                # fallback (rare): use current color cycle
                fc = ax._get_lines.get_next_color()
                # normalize to RGBA
                fc = mpl.colors.to_rgba(fc)

            r, g, b, a = fc
            # adjust alpha for the textbox so it’s legible
            fc_box = (r, g, b, bbox_alpha)

            # choose black/white text for contrast
            txt_color = "black" if _luminance(r, g, b) > 0.6 else "white"

            ax.text(
                x_center,
                med * y_nudge_factor,
                fmt.format(med),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=txt_color,
                bbox=dict(boxstyle="round,pad=0.2", fc=fc_box, ec="none"),
                zorder=100,
                clip_on=False,
            )


# ---------- helpers: median ordering + median labeler (single-category boxplot) ----------
def order_by_median_dict(d, reverse=True):
    """Return (ordered_task_names, stats_dict) where stats_dict[task]=(median, std)."""
    stats = {t: (np.median(vals), np.std(vals)) for t, vals in d.items() if len(vals) > 0}
    ordered = sorted(stats.keys(), key=lambda t: stats[t][0], reverse=reverse)
    return ordered, stats


def annotate_medians_single_boxplot(
    ax, df_long, x_col, y_col, order, fmt="{:.2f}", box_alpha=0.90
):
    """
    Annotate the median for each category on a seaborn.boxplot *without hue*.
    Places the number at the geometric center of each box, using the box facecolor for the label bg.
    Call this AFTER setting any y-limits (so the nudge uses final limits).
    """

    # compute medians in plotting order
    med = df_long.groupby(x_col)[y_col].median().reindex(order)

    # collect PathPatches for boxes (artists in most seaborn versions; fallback to patches)
    patches = [
        p for p in getattr(ax, "artists", []) if isinstance(p, mpl.patches.PathPatch)
    ]
    if not patches:
        patches = [p for p in ax.patches if isinstance(p, mpl.patches.PathPatch)]

    n = min(len(patches), len(order))
    ymin, ymax = ax.get_ylim()
    dy = 0.02 * (ymax - ymin)  # small additive nudge in data units

    for k in range(n):
        patch = patches[k]
        verts = patch.get_path().vertices
        xs, _ = verts[:, 0], verts[:, 1]
        x_center = 0.5 * (xs.min() + xs.max())

        m = med.iloc[k]
        if not np.isfinite(m):
            continue

        # label background color = box facecolor (match legend/palette)
        fc = patch.get_facecolor()
        if fc is None or len(fc) < 3:
            fc = mpl.colors.to_rgba("white", box_alpha)
        else:
            fc = (fc[0], fc[1], fc[2], box_alpha)

        # text color for contrast (simple luminance check)
        lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
        txt_color = "black" if lum > 0.6 else "white"

        # keep label inside the axis (avoid hitting the top bound)
        y_text = min(m + dy, ymax - 0.01 * (ymax - ymin))

        ax.text(
            x_center,
            y_text,
            fmt.format(m),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=txt_color,
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec="none"),
            zorder=100,
            clip_on=False,
        )


###################### task_presence_binarization ######################

###################### dfc_visualization ######################


def _window_indices(
    trs, window_len=8, center="middle", center_time=None, center_index=None, interval=None
):
    T = len(trs)
    trs = np.asarray(trs)
    if interval is not None:
        t0, t1 = interval
        idxs = np.where((trs >= t0) & (trs <= t1))[0]
        if len(idxs) == 0:
            raise ValueError("interval produced no indices; check units.")
        return idxs
    if center_index is not None:
        c = int(np.clip(center_index, 0, T - 1))
    elif center_time is not None:
        c = int(np.argmin(np.abs(trs - center_time)))
    else:
        c = (T - 1) // 2
    half = window_len // 2
    start = max(0, c - half)
    end = min(T, start + window_len)
    start = max(0, end - window_len)
    return np.arange(start, end, dtype=int)


def _common_limits(dfc_dict, robust_percentile=(2, 98), symmetric=True):
    vals = []
    for A in dfc_dict.values():
        R = A.shape[1]
        iu = np.triu_indices(R, 1)
        vals.append(A[:, iu[0], iu[1]].ravel())
    lo, hi = np.percentile(np.concatenate(vals), robust_percentile)
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def figure_dfc_matrices_window_png(
    dfc_dict,
    trs,
    window_len=8,
    center="middle",
    center_time=None,
    center_index=None,
    interval=None,
    cmap="coolwarm",
    outfile="fig_dfc_window.png",
    show_region_ticks=False,
    region_labels=None,
    draw_network_bounds=None,
    dpi=600,
    transparent=False,
    # style knobs
    method_label_size=11,
    tr_label_size=10,
    cbar_label_size=11,
    rotate_method_labels=90,
    method_label_pad=18,  # << controls distance between method names and images
    wspace=None,  # << override column spacing if needed (None = auto)
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": tr_label_size,
            "axes.labelsize": method_label_size,
        }
    )

    methods = list(dfc_dict.keys())
    R = next(iter(dfc_dict.values())).shape[1]

    idxs = _window_indices(
        trs,
        window_len=window_len,
        center=center,
        center_time=center_time,
        center_index=center_index,
        interval=interval,
    )

    vmin, vmax = _common_limits(dfc_dict, robust_percentile=(2, 98), symmetric=True)
    vmin = 0

    # figure sizing
    col_width = 1.6
    row_height = 1.5
    nrows, ncols = len(methods), len(idxs)

    fig = plt.figure(figsize=((ncols + 0.5) * col_width, nrows * row_height))

    # spacing
    auto_wspace = min(0.35, 0.12 + 0.01 * ncols)
    wspace = auto_wspace if wspace is None else wspace
    hspace = 0.25

    # add a dedicated colorbar column on the far right
    gs = gridspec.GridSpec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.06],  # last slot = colorbar
        hspace=hspace,
        wspace=wspace,
    )

    last_im = None
    for r, m in enumerate(methods):
        A = dfc_dict[m]
        for c, t_idx in enumerate(idxs):
            ax = fig.add_subplot(gs[r, c])
            M = A[t_idx].copy()
            np.fill_diagonal(M, np.nan)
            im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")
            last_im = im

            if draw_network_bounds:
                for b in draw_network_bounds:
                    ax.axhline(b - 0.5, linewidth=0.4, color="k")
                    ax.axvline(b - 0.5, linewidth=0.4, color="k")

            if show_region_ticks and region_labels is not None:
                step = max(1, R // 16)
                ticks = np.arange(0, R, step)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(
                    [region_labels[i] for i in ticks], rotation=90, fontsize=6
                )
                ax.set_yticklabels([region_labels[i] for i in ticks], fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if r == 0:
                label = (
                    f"TR{trs[t_idx]}"
                    if np.issubdtype(np.asarray(trs).dtype, np.number)
                    else str(trs[t_idx])
                )
                ax.set_title(label, pad=6, fontsize=tr_label_size, fontweight="bold")

            if c == 0:
                ax.set_ylabel(
                    m,
                    rotation=rotate_method_labels,
                    labelpad=method_label_pad,  # << tighten/loosen here
                    va="center",
                    ha="center",
                    fontsize=method_label_size,
                    fontweight="bold",
                )
                ax.yaxis.set_label_position("left")

    # colorbar in its own axis (no overlap)
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Connectivity", fontsize=cbar_label_size, fontweight="bold")
    cbar.ax.tick_params(labelsize=max(8, cbar_label_size - 1))

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(
        outfile,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
        facecolor="white",
    )
    plt.close(fig)
    print(
        f"Saved {outfile}  |  TR columns: {len(idxs)}  |  vmin={vmin:.3f}, vmax={vmax:.3f}  |  dpi={dpi}"
    )


###################### sample_matrix plots ######################


def nice_step(n, max_ticks=10):
    """Return a 'nice' step (1-2-5x10^k) to keep ≤ max_ticks across [1..n]."""
    if n <= 1:
        return 1
    raw = max(1.0, n / max(2, (max_ticks - 1)))
    exp = np.floor(np.log10(raw))
    frac = raw / (10**exp)
    base = 1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10
    return int(base * (10**exp))


def plot_samples_features(
    X,
    y,
    *,
    sample_order="original",  # "original" | "label" | "label+cluster" | "cluster"
    feature_order="original",  # "original" | "tstat"
    col_order_from_train=None,  # optional np.ndarray (feature indices) to reuse on test
    ZSCORE=True,
    V_RANGE=None,
    cmap="coolwarm",
    title=None,
    save_path=None,
    show=True,
):
    """
    X: (n_samples, n_features) matrix (features in columns)
    y: (n_samples,) binary (0=rest, 1=task)

    Samples are shown along the horizontal axis (time-like), features along the vertical axis.
    If feature_order == "tstat", a slim vertical t-stat bar is shown on the LEFT,
    aligned 1:1 with feature rows (no top t-bar).
    """
    # ---------- prep ----------
    X = np.asarray(X, float)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    # z-score per feature
    Xz = X.copy()
    if ZSCORE:
        mu = Xz.mean(axis=0, keepdims=True)
        sd = Xz.std(axis=0, keepdims=True) + 1e-8
        Xz = (Xz - mu) / sd

    # ---------- feature order ----------
    if feature_order == "tstat":
        if col_order_from_train is not None:
            col_order = np.asarray(col_order_from_train, int)
            t, _ = ttest_ind(Xz[y == 1], Xz[y == 0], axis=0, equal_var=False)
            t_ord = t[col_order]
        else:
            t, _ = ttest_ind(Xz[y == 1], Xz[y == 0], axis=0, equal_var=False)
            col_order = np.argsort(-np.abs(t))  # strongest contrast first
            t_ord = t[col_order]
    else:
        col_order = np.arange(n_features)
        t_ord = None  # no t-stat bar

    # ---------- sample order ----------
    if sample_order == "original":
        row_order = np.arange(n_samples)
        split = np.sum(y == 0)
        draw_separator = False
    elif sample_order == "label":
        rest_idx = np.where(y == 0)[0]
        task_idx = np.where(y == 1)[0]
        row_order = np.r_[rest_idx, task_idx]
        split = len(rest_idx)
        draw_separator = True
    elif sample_order == "label+cluster":

        def order_rows(A):
            if len(A) <= 2:
                return np.arange(len(A))
            return leaves_list(linkage(A, method="average", metric="cosine"))

        rest_idx = np.where(y == 0)[0]
        task_idx = np.where(y == 1)[0]
        rest_order = rest_idx[order_rows(Xz[rest_idx])] if len(rest_idx) else rest_idx
        task_order = task_idx[order_rows(Xz[task_idx])] if len(task_idx) else task_idx
        row_order = np.r_[rest_order, task_order]
        split = len(rest_order)
        draw_separator = True
    elif sample_order == "cluster":

        def order_rows(A):
            if len(A) <= 2:
                return np.arange(len(A))
            return leaves_list(linkage(A, method="average", metric="cosine"))

        all_idx = np.arange(n_samples)
        # rest_order = rest_idx[order_rows(Xz[rest_idx])] if len(rest_idx) else rest_idx
        # task_order = task_idx[order_rows(Xz[task_idx])] if len(task_idx) else task_idx

        row_order = all_idx[order_rows(Xz[all_idx])]
        split = np.sum(y == 0)
        draw_separator = False
    else:
        raise ValueError(
            "sample_order must be one of {'original','label','label+cluster'}"
        )

    # ---------- figure & layout (no top t-bar) ----------
    # W = max(10, min(24, n_samples / 30))
    w_min = 12
    w_max = 24
    width_per_100 = 0.5  # additional width per 100 samples
    W = float(np.clip(w_min + (n_samples / 100.0) * width_per_100, w_min, w_max))
    H = max(6, min(16, n_features / 30))
    fig = plt.figure(figsize=(W, H))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 0.06],  # main heatmap + class strip
        hspace=0.08,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_lab = fig.add_subplot(gs[1, 0])

    # --- VRANGE ---
    if V_RANGE is None:
        Xflat = np.asarray(Xz, float).ravel()
        lo, hi = np.nanpercentile(Xflat, [5, 95])  # robust to outliers; tweak if needed
        V_RANGE = max(abs(lo), abs(hi))  # symmetric around 0 (for diverging cmap)

    # ---------- main heatmap ----------
    img = Xz[row_order, :][:, col_order].T  # (features, samples)
    ax_main.imshow(
        img, aspect="auto", origin="lower", cmap=cmap, vmin=-V_RANGE, vmax=V_RANGE
    )
    n_features = img.shape[0]
    last_idx = n_features - 1

    if n_features < 10:
        # every feature: labels 1..n, positions 0..n-1
        labels_1based = np.arange(1, n_features + 1, dtype=int)
    else:
        step = nice_step(n_features, max_ticks=10)
        # use round multiples of the step
        labels_1based = list(np.arange(step, n_features + 1, step, dtype=int))
        # de-dup & sort (in case step == 1)
        labels_1based = np.unique(labels_1based)

    # convert 1-based labels to 0-based tick positions
    ticks_pos = labels_1based - 1

    # lock y-limits so the last tick isn't clipped
    ax_main.set_ylim(-0.5, last_idx + 0.5)

    # set ticks & labels
    ax_main.set_yticks(ticks_pos)
    ax_main.set_yticklabels([f"{v:d}" for v in labels_1based])
    ax_main.set_ylabel("feature", fontsize=18, fontweight="bold")
    # ax_main.set_xlabel("sample", fontsize=18, fontweight="bold")
    # ax_main.set_xticks([])
    ax_main.tick_params(axis="y", labelsize=18)
    ax_main.tick_params(axis="x", labelsize=18)

    if draw_separator and 0 < split < n_samples:
        ax_main.axvline(split - 0.5, color="k", lw=2)

    # ---------- bottom class strip ----------
    y_reordered = y[row_order]
    cmap_lbl = ListedColormap(
        [[0.85, 0.85, 0.85], [0.25, 0.5, 0.9]]
    )  # rest=gray, task=blue
    ax_lab.imshow(
        y_reordered[None, :], aspect="auto", origin="lower", cmap=cmap_lbl, vmin=0, vmax=1
    )
    ax_lab.set_yticks([])
    ax_lab.set_xticks([])
    # ax_lab.set_title("class", fontsize=11, pad=2)

    # show class labels only when there is label grouping
    if draw_separator:
        n0 = (y_reordered == 0).sum()
        n1 = (y_reordered == 1).sum()
        if n0 > 0:
            x0 = (n0 - 1) / 2.0
            ax_lab.annotate(
                "rest (0)",
                xy=(x0, -0.35),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )
        if n1 > 0:
            x1 = n0 + (n1 - 1) / 2.0
            ax_lab.annotate(
                "task (1)",
                xy=(x1, -0.35),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )

    # --- move the class bar (ax_lab) down a bit ---
    fig.canvas.draw()  # ensure positions are current
    lab_box = ax_lab.get_position()  # [x0, y0, width, height] in figure coords
    down = 0.050  # how much to move down (figure fraction)
    new_y0 = max(0.01, lab_box.y0 - down)  # keep it inside the figure
    ax_lab.set_position([lab_box.x0, new_y0, lab_box.width, lab_box.height])

    # after you position ax_lab (i.e., after ax_lab.set_position([...]))
    ax_lab.xaxis.set_label_position("top")
    ax_lab.set_xlabel("sample", labelpad=6, fontweight="bold", fontsize=18)
    # keep the strip clean
    ax_lab.tick_params(
        axis="x", which="both", length=0, labelbottom=False, labeltop=False
    )

    # (re)grab the updated box for the colorbar placement that comes next
    lab_box = ax_lab.get_position()

    # ---------- LEFT vertical t-stat bar (only if feature_order=="tstat") ----------
    if t_ord is not None:
        fig.canvas.draw()
        main_box = ax_main.get_position()  # figure coords

        tbar_left_width = 0.010  # ~2% fig width
        tbar_left_pad = 0.035 / W * 24  # gap from main heatmap, proportional to fig width

        x0 = max(0.01, main_box.x0 - tbar_left_pad - tbar_left_width)
        y0 = main_box.y0
        w = tbar_left_width
        h = main_box.height

        ax_tleft = fig.add_axes([x0, y0, w, h])
        m = np.nanmax(np.abs(t_ord)) if np.isfinite(t_ord).any() else 1.0
        ax_tleft.imshow(
            t_ord[:, None], origin="lower", aspect="auto", cmap=cmap, vmin=-m, vmax=m
        )
        ax_tleft.set_xticks([])
        ax_tleft.set_yticks([])
        ax_tleft.set_title("t-stat", fontsize=11, pad=2, fontweight="bold")

    if title:
        fig.suptitle(title, y=0.995, fontsize=12, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(row_order=row_order, col_order=col_order)


def save_scalar_colorbar(
    cmap="coolwarm",
    vmin=-2.0,
    vmax=2.0,  # use the same V_RANGE you use in plots
    label="z-scored feature value",
    filename="zscore_colorbar.png",
    orientation="horizontal",
    figsize=(6, 0.4),  # width, height in inches
    dpi=300,
    ticks=None,
):
    """
    Saves a standalone scalar colorbar image you can reuse in the paper.
    """
    # Make a dummy mappable with the correct colormap and limits
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(
        [0.05, 0.35, 0.90, 0.30]
        if orientation == "horizontal"
        else [0.35, 0.05, 0.30, 0.90]
    )

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax, orientation=orientation)
    cb.set_label(label, fontsize=18, fontweight="bold")

    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.tick_params(labelsize=18)

    fig.savefig(filename, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def nearest_neighbor_match(X, y):
    """
    Compute fraction of matching labels for k=1,5,10 nearest neighbors.
    For k>1, this returns fraction of *all neighbor votes* that match the label.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Fit once with max k
    max_k = 10
    nbrs = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric="correlation",
        algorithm="auto",
    ).fit(X)

    # Compute neighbors for all samples once
    indices = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # drop self

    # Labels of all neighbors: shape (N, 10)
    neighbor_labels = y[indices]

    # Expand y to shape (N,1) for vectorized comparison
    y_col = y.reshape(-1, 1)

    # Boolean match matrix: (N,10)
    match_matrix = neighbor_labels == y_col

    # Compute metrics
    match_1 = np.mean(match_matrix[:, :1])  # first neighbor
    match_5 = np.mean(match_matrix[:, :5])  # first 5 neighbors
    match_10 = np.mean(match_matrix[:, :10])  # all 10 neighbors

    return match_1, match_5, match_10


# def other_class_max_corr(X, y, method="fast", metric="cosine"):
#     """
#     Compute max cross-class similarity (Pearson or cosine) per sample and summary stats.

#     Parameters
#     ----------
#     X : array, shape (n_samples, n_features)
#     y : array, shape (n_samples,)
#     method : "slow" or "fast"
#         slow  = loop-based (OOM-safe but slow)
#         fast  = matrix-based (very fast but uses O(N^2) memory)
#     metric : "pearson" or "cosine"
#         pearson = correlation after mean-centering
#         cosine  = correlation without mean-centering

#     Returns
#     -------
#     median, fraction_above_0.9, 95th_percentile, fraction_z_gt_1.645
#     """
#     X = np.asarray(X, dtype=float)
#     y = np.asarray(y)

#     # ======================================================
#     # Helper: compute normalized X depending on metric
#     # ======================================================
#     def normalize(X):
#         if metric == "pearson":
#             Xn = X - X.mean(axis=1, keepdims=True)
#         elif metric == "cosine":
#             Xn = X.copy()
#         else:
#             raise ValueError("metric must be 'pearson' or 'cosine'")

#         norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
#         return Xn / norms

#     # ======================================================
#     # SLOW METHOD (loop-based, safe for large N)
#     # ======================================================
#     if method == "slow":
#         all_corrs = []
#         for i, sample in enumerate(X):
#             class_label = y[i]
#             other_class_label = 1 - class_label

#             # extract opposite-class samples
#             X_other = X[y == other_class_label]

#             # normalize sample and opposite-class samples based on metric
#             s = sample.reshape(1, -1)
#             s_norm = normalize(s)[0]
#             X_other_norm = normalize(X_other)

#             # dot products = cosine or Pearson similarity
#             corrs = X_other_norm @ s_norm

#             all_corrs.append(np.max(corrs))

#         all_corrs = np.asarray(all_corrs)

#     # ======================================================
#     # FAST METHOD (matrix multiplication)
#     # ======================================================
#     elif method == "fast":
#         X_norm = normalize(X)  # normalize entire matrix according to metric

#         # similarity matrix
#         sim_matrix = X_norm @ X_norm.T  # (N × N)

#         # mask for opposite-class pairs
#         y = y.astype(int)
#         other_mask = y[:, None] != y[None, :]

#         # mask same-class and diagonal
#         cross_sims = np.where(other_mask, sim_matrix, -np.inf)

#         # max similarity to opposite class
#         all_corrs = np.max(cross_sims, axis=1)

#     else:
#         raise ValueError("method must be 'slow' or 'fast'")

#     # ======================================================
#     # Summary statistics
#     # ======================================================
#     median = np.median(all_corrs)
#     above_90 = np.mean(all_corrs > 0.9)
#     percentile_95 = np.percentile(all_corrs, 95)

#     mean_corr = all_corrs.mean()
#     std_corr = all_corrs.std(ddof=1)
#     z = (all_corrs - mean_corr) / (std_corr + 1e-12)
#     high_frac = np.mean(z > 1.645) * 100

#     return median, above_90, percentile_95, high_frac


def other_class_corr(X, y, method="fast", metric="cosine"):
    """
    Compute cross-class similarity (Pearson or cosine) for ALL opposite-class pairs
    (each pair counted only once) and summarize their distribution.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,)
    method : "slow" or "fast"
        slow  = loop-based (avoids full N×N matrix, but slower)
        fast  = matrix-based (computes class0×class1 block)
    metric : "pearson" or "cosine"
        pearson = correlation after mean-centering each row
        cosine  = cosine similarity without mean-centering

    Returns
    -------
    fraction_above_0.9, fraction_above_0.95, fraction_above_0.99, fraction_above_0.999
    0.0–1.0 fractions over ALL unique cross-class similarities.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # ------------------------------------------------------
    # Helper: row-wise normalization
    # ------------------------------------------------------
    def normalize_rows(X_):
        if metric == "pearson":
            Xn = X_ - X_.mean(axis=1, keepdims=True)
        elif metric == "cosine":
            Xn = X_.copy()
        else:
            raise ValueError("metric must be 'pearson' or 'cosine'")

        norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
        return Xn / norms

    # Binary labels assumed: 0/1 (or two unique values)
    unique_labels = np.unique(y)
    if unique_labels.size != 2:
        raise ValueError("This function assumes exactly two classes.")

    label_a, label_b = unique_labels

    idx_a = np.where(y == label_a)[0]
    idx_b = np.where(y == label_b)[0]

    # ------------------------------------------------------
    # SLOW METHOD (loop over one class only)
    # ------------------------------------------------------
    if method == "slow":
        all_corrs = []

        X_b = X[idx_b]
        X_b_norm = normalize_rows(X_b)

        for i in idx_a:
            s = X[i].reshape(1, -1)
            s_norm = normalize_rows(s)[0]

            # similarities between this sample and ALL samples in other class
            corrs = X_b_norm @ s_norm  # shape: (n_b,)

            all_corrs.extend(corrs.tolist())

        all_corrs = np.asarray(all_corrs)

    # ------------------------------------------------------
    # FAST METHOD (block matrix classA × classB)
    # ------------------------------------------------------
    elif method == "fast":
        X_norm = normalize_rows(X)

        Xa = X_norm[idx_a]  # (n_a, F)
        Xb = X_norm[idx_b]  # (n_b, F)

        # similarities for all unique cross-class pairs (label_a vs label_b)
        # shape: (n_a, n_b)
        sim_block = Xa @ Xb.T

        # flatten: each pair counted exactly once
        all_corrs = sim_block.ravel()

    else:
        raise ValueError("method must be 'slow' or 'fast'")

    # ------------------------------------------------------
    # Summary statistics over ALL unique cross-class similarities
    # ------------------------------------------------------
    above_90 = np.mean(all_corrs > 0.9)
    above_95 = np.mean(all_corrs > 0.95)
    above_99 = np.mean(all_corrs > 0.99)
    above_999 = np.mean(all_corrs > 0.999)

    return above_90, above_95, above_99, above_999


from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold


def ldc_crossvalidated(X, y, n_splits=2, random_state=None):
    """
    Cross-validated Mahalanobis distance (LDC) between two classes.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix. Rows are samples, columns are features.
    y : array-like, shape (n_samples,)
        Class labels. Must contain exactly TWO unique labels.
    n_splits : int, default=2
        Number of CV splits (partitions). 2 is the classic LDC setup.
    random_state : int or None
        Seed for reproducible splits.

    Returns
    -------
    ldc : float
        Cross-validated Mahalanobis distance (LDC).
        Unbiased estimate of squared distance between class means
        in noise-whitened space (optionally divided by n_features).
    pairwise_ldcs : np.ndarray, shape (n_pairs,)
        LDC values for each fold-pair used in the averaging.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # ----- 1) Check binary labels -----
    unique_labels = np.unique(y)
    if unique_labels.size != 2:
        raise ValueError("ldc_crossvalidated currently assumes exactly two classes.")
    label_a, label_b = unique_labels

    n_samples, n_features = X.shape

    # ----- 2) Estimate noise covariance (within-class) -----
    # Subtract class means to get residuals, then estimate covariance on residuals.
    X_resid = X.copy()
    for lbl in unique_labels:
        mask = y == lbl
        X_resid[mask] -= X_resid[mask].mean(axis=0, keepdims=True)

    # Shrinkage covariance for stability in high-dim / low-n regimes
    lw = LedoitWolf().fit(X_resid)
    Sigma = lw.covariance_

    # ----- 3) Compute whitening transform W such that Sigma^{-1} = W^T W -----
    # Using Cholesky: Sigma = L L^T  =>  Sigma^{-1} = L^{-T} L^{-1}
    # Choose W = L^{-1}; then W^T W = Sigma^{-1}.
    L = np.linalg.cholesky(Sigma)
    # Solve L * M = I for M, then W = M
    W = np.linalg.inv(L)  # (n_features x n_features)

    # Whitened data
    Xw = X @ W  # each row is whitened pattern

    # ----- 4) Cross-validation splits -----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    deltas = []  # mean-difference vectors per fold in whitened space

    for train_idx, _ in skf.split(Xw, y):
        X_fold = Xw[train_idx]
        y_fold = y[train_idx]

        # class means in this fold
        mu_a = X_fold[y_fold == label_a].mean(axis=0)
        mu_b = X_fold[y_fold == label_b].mean(axis=0)

        delta = mu_a - mu_b  # difference of class means
        deltas.append(delta)

    deltas = np.vstack(deltas)  # shape (n_splits, n_features)

    # ----- 5) Cross-validated Mahalanobis distance (LDC) -----
    # For each pair of independent partitions f != g:
    #   LDC_fg = delta_f^T delta_g / n_features
    # Average over all unique pairs.
    pairwise_ldcs = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            ldc_ij = np.dot(deltas[i], deltas[j]) / n_features
            pairwise_ldcs.append(ldc_ij)

    pairwise_ldcs = np.asarray(pairwise_ldcs)
    ldc = pairwise_ldcs.mean()

    return ldc
