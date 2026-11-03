"""
Classification of the cycle-count models as described cards.

  Rows    = where the model's knowledge comes from
  Columns = analytical (predicts from counts) / simulation (runs the program)
  Arrows  = "built on" relations between models

Edit ROWS, MODELS and ARROWS, then re-run:
    python3 figure_scripts/model_cards.py
Outputs figures/model_cards.pdf (for LaTeX), .svg and .png.
Formulas use matplotlib mathtext (LaTeX subset) between $...$.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# 1. Content
# ---------------------------------------------------------------------------
ROWS = [  # key, label, colour
    ("mech",   "Mechanistic",           "#2a78d6"),
    ("hybrid", "Mechanistic-empirical", "#1baf7a"),
    ("emp",    "Empirical",             "#eb6834"),
]
COLUMNS = [  # key, title, subtitle, number of card slots
    ("analytical", "Analytical", "predicts from counts of the program", 2),
    ("simulation", "Simulation", "executes the program cycle by cycle", 1),
]

# slot: position inside the column (0, 1, ...); 0.5 = centred across two slots
MODELS = [
    dict(key="mech", name="Mechanistic", row="mech", col="analytical", slot=0.5,
         formula=r"$\Sigma_i\; n_i\, c_i$",
         desc="Manual's cycle cost of each\ninstruction, summed.",
         params="0 fitted"),
    dict(key="veri", name="Verilator model", row="mech", col="simulation", slot=0,
         formula=r"RTL $\rightarrow$ C++",
         desc="Processor RTL run cycle by\ncycle; exact reference.",
         params="0 fitted"),
    dict(key="mols", name="Mechanistic OLS", row="hybrid", col="analytical", slot=0,
         formula=r"$a \cdot \mathrm{mech} + b$",
         desc="Manual's estimate with a\nfitted slope and intercept.",
         params="2 fitted"),
    dict(key="memp", name="Mechanistic-empirical", row="hybrid", col="analytical", slot=1,
         formula=r"$\mathrm{mech} + m\,n_{\mathrm{mem}} + \Sigma_{j}\, f_j\,n_{\mathrm{call},j} + b_{\mathrm{team}}$",
         desc="Manual's estimate, fitted memory and\ncall costs, team intercept.",
         params="8 fitted", formula_size=7.3),
    dict(key="icols", name="Instruction count OLS", row="emp", col="analytical", slot=0,
         formula=r"$a \cdot n_{\mathrm{instr}} + b$",
         desc="Instruction count with a\nfitted slope and intercept.",
         params="2 fitted"),
    dict(key="lasso", name="Lasso", row="emp", col="analytical", slot=1,
         formula=r"$\Sigma_{k\,\in\,\{\mathrm{ADD,\,BR,\,RAW,\,SW}\}}\; x_k\, n_k$",
         desc="Sparse regression on\noperation and hazard counts.",
         params="4 fitted"),
]

ARROWS = [("mech", "mols"), ("mech", "memp")]   # (from, to)

# One colour per model, shared with the other figures of the thesis.
MODEL_COLORS = {
    "mechanistic ols":       "#2a78d6",   # slot 1, blue
    "mechanistic-empirical": "#eb6834",   # slot 2, orange
    "lasso":                 "#1baf7a",   # slot 3, aqua
    "instruction count ols": "#eda100",   # slot 4, yellow
    "mechanistic":           "#e87ba4",   # slot 5, magenta
    "verilator model":       "#8a8984",   # grey (reference)
}

FIG_SIZE = (6.3, 4.6)   # inches; 6.3 in ~ 16 cm text width
FONT_SIZE = 8
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"   # repo-level figures/

# ---------------------------------------------------------------------------
# 2. Geometry & drawing (no need to touch below for normal edits)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#d6d5d0"
CARD_EDGE = "#dcdbd6"
SURFACE = "#ffffff"

LABEL_W = 0.6     # left band with the rotated row labels
SLOT_W = 3.15      # card width
GAP_X = 0.28       # horizontal gap between slots / columns
COL_GAP = 0.45     # extra gap before a new column
HEADER_H = 0.85
CARD_H = 1.72
ROW_PAD = 0.26


def use_thesis_font():
    found = None
    for path in font_manager.findSystemFonts(fontext="otf") + font_manager.findSystemFonts():
        if "lmroman10-" in path.lower():
            font_manager.fontManager.addfont(path)
            found = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["font.family"] = found or "serif"
    plt.rcParams["mathtext.fontset"] = "cm"


def tint(hex_color, amount):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


# --- text fitting: every text is measured and shrunk / re-wrapped so it stays
#     inside its card, whatever font the machine running the script has ---
MIN_FONT = 5.5


def _width(ax, t):
    """Width of a text object in data units."""
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(r)
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(bb.x0, 0), (bb.x1, 0)])
    return abs(x1 - x0)


def fit_line(ax, t, max_w):
    """Shrink a one-line text until it is at most max_w wide."""
    while _width(ax, t) > max_w and t.get_fontsize() > MIN_FONT:
        t.set_fontsize(t.get_fontsize() - 0.25)


def fit_paragraph(ax, t, text, max_w, max_lines, last_line_w=None):
    """Greedy word-wrap: lines up to max_w wide, the last allowed line up to
    last_line_w (room for the badge). Shrinks the font if it needs more lines."""
    last_line_w = max_w if last_line_w is None else last_line_w
    words = text.replace("\n", " ").split()
    while True:
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            t.set_text(trial)
            limit = last_line_w if len(lines) == max_lines - 1 else max_w
            if cur and _width(ax, t) > limit:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        t.set_text(lines[-1])
        last_ok = len(lines) < max_lines or _width(ax, t) <= last_line_w
        t.set_text("\n".join(lines))
        if (len(lines) <= max_lines and last_ok) or t.get_fontsize() <= MIN_FONT:
            return
        t.set_fontsize(t.get_fontsize() - 0.25)


def main():
    use_thesis_font()
    plt.rcParams.update({"font.size": FONT_SIZE, "pdf.fonttype": 42, "svg.fonttype": "path"})

    # column x-origins
    col_x, x = {}, LABEL_W + 0.2
    for i, (ckey, _, _, slots) in enumerate(COLUMNS):
        if i:
            x += COL_GAP
        col_x[ckey] = (x, slots)
        x += slots * SLOT_W + (slots - 1) * GAP_X
        x += GAP_X if i < len(COLUMNS) - 1 else 0
    total_w = x + 0.05
    row_h = CARD_H + 2 * ROW_PAD
    total_h = HEADER_H + row_h * len(ROWS)
    row_y = {rkey: HEADER_H + i * row_h for i, (rkey, *_) in enumerate(ROWS)}
    row_col = {rkey: c for rkey, _, c in ROWS}

    fig = plt.figure(figsize=FIG_SIZE, facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_axis_off()

    # column headers + separator
    for i, (ckey, title, sub, slots) in enumerate(COLUMNS):
        x0, n = col_x[ckey]
        cx = x0 + (n * SLOT_W + (n - 1) * GAP_X) / 2
        ax.text(cx, HEADER_H * 0.36, title, ha="center", va="center",
                fontsize=FONT_SIZE + 2, color=INK)
        sub_t = ax.text(cx, HEADER_H * 0.72, sub, ha="center", va="center",
                        fontsize=FONT_SIZE - 0.5, color=INK_2)
        fit_line(ax, sub_t, n * SLOT_W + (n - 1) * GAP_X)
        if i:
            sx = x0 - (COL_GAP + GAP_X) / 2
            ax.plot([sx, sx], [HEADER_H + 0.1, total_h - 0.05], color=RULE, lw=0.8,
                    linestyle=(0, (3, 3)))

    # row bands: rotated label + light rule
    for i, (rkey, label, color) in enumerate(ROWS):
        y0 = row_y[rkey]
        ax.text(0.3, y0 + row_h / 2, label.replace("-", "-\n") if len(label) > 14 else label,
                rotation=90, ha="center", va="center", color=INK,
                fontsize=FONT_SIZE + 0.5, linespacing=1.1)
        ax.plot([0, total_w], [y0, y0], color=INK_2 if i == 0 else RULE,
                lw=0.9 if i == 0 else 0.6)
    ax.plot([0, total_w], [total_h, total_h], color=RULE, lw=0.6)

    # cards
    boxes = {}
    for m in MODELS:
        x0, _ = col_x[m["col"]]
        cx0 = x0 + m["slot"] * (SLOT_W + GAP_X)
        cy0 = row_y[m["row"]] + ROW_PAD
        color = MODEL_COLORS[m["name"].lower()]
        boxes[m["key"]] = (cx0, cy0, SLOT_W, CARD_H)

        ax.add_patch(FancyBboxPatch((cx0, cy0), SLOT_W, CARD_H,
                                    boxstyle="round,pad=0,rounding_size=0.09",
                                    facecolor=SURFACE, edgecolor=CARD_EDGE, lw=0.8, zorder=2))
        ax.add_patch(FancyBboxPatch((cx0, cy0), SLOT_W, 0.09,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=color, edgecolor="none", zorder=3))
        pad = 0.17
        inner_w = SLOT_W - 2 * pad
        title = ax.text(cx0 + pad, cy0 + 0.34, m["name"], ha="left", va="center",
                        fontsize=FONT_SIZE + 1.5, color=INK, zorder=4)
        fit_line(ax, title, inner_w)
        badge = ax.text(cx0 + SLOT_W - pad, cy0 + CARD_H - 0.24, m["params"], ha="right",
                        va="center", fontsize=FONT_SIZE - 1.5, color=INK_2, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6",
                                  facecolor=tint(color, 0.86), edgecolor="none"))
        formula = ax.text(cx0 + SLOT_W / 2, cy0 + 0.80, m["formula"], ha="center",
                          va="center", fontsize=m.get("formula_size", FONT_SIZE),
                          color=INK, zorder=4)
        fit_line(ax, formula, inner_w)
        # description: keep clear of the badge on the right
        badge_w = _width(ax, badge) + 0.18
        desc = ax.text(cx0 + pad, cy0 + 1.14, "", ha="left", va="top",
                       fontsize=FONT_SIZE - 0.8, color=INK_2, linespacing=1.3, zorder=4)
        fit_paragraph(ax, desc, m["desc"], inner_w, max_lines=2,
                      last_line_w=inner_w - badge_w)

    # "built on" arrows
    for src, dst in ARROWS:
        sx, sy, sw, sh = boxes[src]
        dx, dy, dw, dh = boxes[dst]
        start = (sx + sw / 2 + (dx + dw / 2 - (sx + sw / 2)) * 0.35, sy + sh)
        end = (dx + dw / 2, dy)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     color=INK_2, lw=0.9, zorder=1,
                                     connectionstyle="arc3,rad=0"))

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf",): #"svg", "png"):
        fig.savefig(FIG_DIR / f"model_cards.{ext}", dpi=240, bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'model_cards'}.pdf")# / .svg / .png")


if __name__ == "__main__":
    main()
