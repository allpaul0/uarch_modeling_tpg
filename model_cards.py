"""
Classification of the cycle-count models as described cards.

  Rows    = where the model's knowledge comes from
  Columns = analytical (predicts from counts) / simulation (runs the program)
  Arrows  = "built on" relations between models

Edit ROWS, MODELS and ARROWS, then re-run:
    python3 model_cards.py
Outputs model_cards.pdf (for LaTeX), .svg and .png.
Formulas use matplotlib mathtext (LaTeX subset) between $...$.
"""

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
         desc="The processor's RTL, run cycle\nby cycle. Exact reference.",
         params="0 fitted"),
    dict(key="mols", name="Mechanistic OLS", row="hybrid", col="analytical", slot=0,
         formula=r"$a \cdot \mathrm{mech} + b$",
         desc="Manual's estimate, rescaled\nto measurements.",
         params="2 fitted"),
    dict(key="memp", name="Mechanistic-empirical", row="hybrid", col="analytical", slot=1,
         formula=r"$\mathrm{mech}' + m\,n_{\mathrm{mem}} + \Sigma_{j}\, f_j\,n_{\mathrm{call},j} + t$",
         desc="Manual's estimate plus fitted\ncosts for memory, calls, team.",
         params="8 fitted"),
    dict(key="icols", name="Instruction count OLS", row="emp", col="analytical", slot=0,
         formula=r"$a \cdot n_{\mathrm{instr}} + b$",
         desc="Number of instructions,\ntimes a fitted cost.",
         params="2 fitted"),
    dict(key="lasso", name="Lasso", row="emp", col="analytical", slot=1,
         formula=r"$\Sigma_{k\,\in\,\{\mathrm{ADD,\,MUL,\,RAW,\,SW}\}}\; x_k\, n_k$",
         desc="Sparse regression on\noperation and hazard counts.",
         params="4 fitted"),
]

ARROWS = [("mech", "mols"), ("mech", "memp")]   # (from, to)

FIG_SIZE = (6.3, 4.4)   # inches; 6.3 in ~ 16 cm text width
FONT_SIZE = 8

# ---------------------------------------------------------------------------
# 2. Geometry & drawing (no need to touch below for normal edits)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#d6d5d0"
CARD_EDGE = "#dcdbd6"
SURFACE = "#ffffff"

LABEL_W = 0.7     # left band with the rotated row labels
SLOT_W = 3.15      # card width
GAP_X = 0.28       # horizontal gap between slots / columns
COL_GAP = 0.45     # extra gap before a new column
HEADER_H = 0.85
CARD_H = 1.6
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


def main():
    use_thesis_font()
    plt.rcParams.update({"font.size": FONT_SIZE, "pdf.fonttype": 42, "svg.fonttype": "none"})

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

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=SURFACE)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_axis_off()

    # column headers + separator
    for i, (ckey, title, sub, slots) in enumerate(COLUMNS):
        x0, n = col_x[ckey]
        cx = x0 + (n * SLOT_W + (n - 1) * GAP_X) / 2
        ax.text(cx, HEADER_H * 0.36, title, ha="center", va="center",
                fontsize=FONT_SIZE + 2, color=INK)
        ax.text(cx, HEADER_H * 0.72, sub, ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, color=INK_2)
        if i:
            sx = x0 - (COL_GAP + GAP_X) / 2
            ax.plot([sx, sx], [HEADER_H + 0.1, total_h - 0.05], color=RULE, lw=0.8,
                    linestyle=(0, (3, 3)))

    # row bands: rotated label + light rule
    for i, (rkey, label, color) in enumerate(ROWS):
        y0 = row_y[rkey]
        ax.add_patch(Rectangle((0.05, y0 + 0.12), 0.07, row_h - 0.24, color=color, lw=0))
        ax.text(0.42, y0 + row_h / 2, label.replace("-", "-\n") if len(label) > 14 else label,
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
        color = row_col[m["row"]]
        boxes[m["key"]] = (cx0, cy0, SLOT_W, CARD_H)

        ax.add_patch(FancyBboxPatch((cx0, cy0), SLOT_W, CARD_H,
                                    boxstyle="round,pad=0,rounding_size=0.09",
                                    facecolor=SURFACE, edgecolor=CARD_EDGE, lw=0.8, zorder=2))
        ax.add_patch(FancyBboxPatch((cx0, cy0), SLOT_W, 0.09,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=color, edgecolor="none", zorder=3))
        pad = 0.17
        ax.text(cx0 + pad, cy0 + 0.34, m["name"], ha="left", va="center",
                fontsize=FONT_SIZE + 1.5, color=INK, zorder=4)
        ax.text(cx0 + SLOT_W - pad, cy0 + 0.34, m["params"], ha="right", va="center",
                fontsize=FONT_SIZE - 1.5, color=INK_2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6",
                          facecolor=tint(color, 0.86), edgecolor="none"))
        ax.text(cx0 + SLOT_W / 2, cy0 + 0.80, m["formula"], ha="center", va="center",
                fontsize=m.get("formula_size", FONT_SIZE), color=INK, zorder=4)
        ax.text(cx0 + pad, cy0 + 1.14, m["desc"], ha="left", va="top",
                fontsize=FONT_SIZE - 0.8, color=INK_2, linespacing=1.3, zorder=4)

    # "built on" arrows
    for src, dst in ARROWS:
        sx, sy, sw, sh = boxes[src]
        dx, dy, dw, dh = boxes[dst]
        start = (sx + sw / 2 + (dx + dw / 2 - (sx + sw / 2)) * 0.35, sy + sh)
        end = (dx + dw / 2, dy)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     color=INK_2, lw=0.9, zorder=1,
                                     connectionstyle="arc3,rad=0"))

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"figures/model_cards.{ext}", dpi=240, bbox_inches="tight")
    print("wrote model_cards.pdf / .svg / .png")


if __name__ == "__main__":
    main()
