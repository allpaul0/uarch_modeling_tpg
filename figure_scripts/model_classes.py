"""
Classification of the cycle-count models (no numeric axes).

  Rows    = where the model's knowledge comes from
            (mechanistic / mechanistic-empirical / empirical)
  Columns = how a prediction is obtained
            (analytical: from counts of the program, without running it /
             simulation: by executing the program cycle by cycle)

Edit ROWS, COLUMNS and MODELS, then re-run:
    python3 figure_scripts/model_classes.py
Outputs figures/model_classes.pdf (for LaTeX), .svg and .png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# 1. Content
# ---------------------------------------------------------------------------
ROWS = [  # key, title, one-line definition, colour
    ("mech",   "Mechanistic",           "built from the processor's\ndocumented behaviour", "#2a78d6"),
    ("hybrid", "Mechanistic-empirical", "mechanistic structure,\nfitted coefficients",       "#1baf7a"),
    ("emp",    "Empirical",             "relationship learned\nfrom measurements",          "#eb6834"),
]
COLUMNS = [  # key, title, one-line definition
    ("analytical", "Analytical", "predicts from counts of the program"),
    ("simulation", "Simulation", "executes the program cycle by cycle"),
]

# name, row, column, subtitle (shown under the name; "" for none)
MODELS = [
    ("Mechanistic",           "mech",   "analytical", "no fitted parameters"),
    ("Verilator model",       "mech",   "simulation", "no fitted parameters"),
    ("Mechanistic OLS",       "hybrid", "analytical", "2 fitted parameters"),
    ("Mechanistic-empirical", "hybrid", "analytical", "8 fitted parameters"),
    ("Instruction count OLS", "emp",    "analytical", "2 fitted parameters"),
    ("Lasso",                 "emp",    "analytical", "one per feature kept by Lasso"),
]

EMPTY_CELL_TEXT = "—"     # shown in cells with no model; "" to leave blank
FIG_SIZE = (6.3, 4.6)          # inches; 6.3 in ~ 16 cm text width
FONT_SIZE = 9.5
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"   # repo-level figures/

# ---------------------------------------------------------------------------
# 2. Drawing (no need to touch below for normal edits)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#9a9993"
RULE = "#d6d5d0"
SURFACE = "#ffffff"

HEADER_W = 2.1     # width of the row-header column (data units)
COL_W = 3.0        # width of each model column
HEADER_H = 0.75    # height of the column-header band
CHIP_H = 0.62      # height of one model chip
CHIP_GAP = 0.12
CELL_PAD = 0.16


def use_thesis_font():
    found = None
    for path in font_manager.findSystemFonts(fontext="otf") + font_manager.findSystemFonts():
        if "lmroman10-" in path.lower():
            font_manager.fontManager.addfont(path)
            found = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["font.family"] = found or "serif"


def tint(hex_color, amount):
    """Mix a colour with white (amount 0 = colour, 1 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def main():
    use_thesis_font()
    plt.rcParams.update({"font.size": FONT_SIZE, "pdf.fonttype": 42, "svg.fonttype": "none"})

    # row heights follow the fullest cell in each row
    def cell_models(r, c):
        return [m for m in MODELS if m[1] == r and m[2] == c]

    row_h = {}
    for rkey, *_ in ROWS:
        n = max(1, max(len(cell_models(rkey, ckey)) for ckey, *_ in COLUMNS))
        row_h[rkey] = 2 * CELL_PAD + n * CHIP_H + (n - 1) * CHIP_GAP

    total_w = HEADER_W + COL_W * len(COLUMNS)
    total_h = HEADER_H + sum(row_h.values())

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=SURFACE)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)          # y grows downwards
    ax.set_axis_off()

    # column headers
    for i, (ckey, title, desc) in enumerate(COLUMNS):
        cx = HEADER_W + COL_W * (i + 0.5)
        ax.text(cx, HEADER_H * 0.38, title, ha="center", va="center",
                fontsize=FONT_SIZE + 1, color=INK)
        ax.text(cx, HEADER_H * 0.75, desc, ha="center", va="center",
                fontsize=FONT_SIZE - 1, color=INK_2)

    # rows
    y = HEADER_H
    ax.plot([0, total_w], [y, y], color=INK_2, lw=0.9)
    for rkey, title, desc, color in ROWS:
        h = row_h[rkey]
        # row header with colour bar
        ax.add_patch(plt.Rectangle((0, y + 0.1), 0.06, h - 0.2, color=color, lw=0))
        ax.text(0.18, y + h / 2 - 0.06, title, ha="left", va="bottom",
                fontsize=FONT_SIZE + 0.5, color=INK)
        ax.text(0.18, y + h / 2 + 0.04, desc, ha="left", va="top",
                fontsize=FONT_SIZE - 1.5, color=INK_2, linespacing=1.15)

        for i, (ckey, *_rest) in enumerate(COLUMNS):
            x0 = HEADER_W + COL_W * i
            models = cell_models(rkey, ckey)
            if not models:
                ax.text(x0 + COL_W / 2, y + h / 2, EMPTY_CELL_TEXT, ha="center",
                        va="center", color=INK_3)
                continue
            block_h = len(models) * CHIP_H + (len(models) - 1) * CHIP_GAP
            cy = y + (h - block_h) / 2
            for name, _, _, sub in models:
                chip = FancyBboxPatch((x0 + 0.25, cy), COL_W - 0.5, CHIP_H,
                                      boxstyle="round,pad=0,rounding_size=0.08",
                                      facecolor=tint(color, 0.86), edgecolor=color, lw=0.9)
                ax.add_patch(chip)
                if sub:
                    ax.text(x0 + COL_W / 2, cy + CHIP_H * 0.37, name, ha="center",
                            va="center", color=INK)
                    ax.text(x0 + COL_W / 2, cy + CHIP_H * 0.70, sub, ha="center",
                            va="center", color=INK_2, fontsize=FONT_SIZE - 1.5)
                else:
                    ax.text(x0 + COL_W / 2, cy + CHIP_H / 2, name, ha="center",
                            va="center", color=INK)
                cy += CHIP_H + CHIP_GAP

        y += h
        ax.plot([0, total_w], [y, y], color=RULE, lw=0.7)

    # column separators
    for i in range(len(COLUMNS)):
        x = HEADER_W + COL_W * i
        ax.plot([x, x], [HEADER_H, total_h], color=RULE, lw=0.7)

    fig.tight_layout(pad=0.2)
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf",): #, "svg", "png"):
        fig.savefig(FIG_DIR / f"model_classes.{ext}", dpi=220, bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'model_classes'}.pdf") #/ .svg / .png")


if __name__ == "__main__":
    main()