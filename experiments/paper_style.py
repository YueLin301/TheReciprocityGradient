"""Shared color palette for all paper figures.
Primary: RGB(191,0,64) = LaTeX purple
Contrast: RGB(0,111,185) = LaTeX NavyBlue, RGB(57,127,50) = LaTeX OliveGreen
Two heatmap palettes provided by user."""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Three core colors (matching LaTeX named colors) ===
PRIMARY = '#BF0040'       # purple  — RGB(191, 0, 64)
NAVY = '#006FB9'          # NavyBlue — RGB(0, 111, 185)
OLIVE = '#397F32'         # OliveGreen — RGB(57, 127, 50)

PRIMARY_LIGHT = '#E88098'
PRIMARY_DARK = '#7F002A'
NAVY_LIGHT = '#6CB4E0'
OLIVE_LIGHT = '#7FBF78'

SLATE = '#455A64'         # neutral dark
C_FLAT = '#BDBDBD'        # gray for flat seeds

# === Heatmap 1 (purple-rose) ===
HM1 = ['#EBCCE2','#DDA1BE','#C87D98','#B25F79','#9B3F5C','#832440']
CMAP_HM1 = LinearSegmentedColormap.from_list('hm1',
    [matplotlib.colors.to_rgb(c) for c in HM1], N=256)

# === Heatmap 2 (warm peach-red) ===
HM2 = ['#FEE3CE','#EABAA1','#DC917B','#D16D5B','#C44438','#B7282E']
CMAP_HM2 = LinearSegmentedColormap.from_list('hm2',
    [matplotlib.colors.to_rgb(c) for c in HM2], N=256)

# Diverging: red ← white → blue
_div = [HM2[-1], HM2[2], '#FFFFFF', NAVY_LIGHT, NAVY]
CMAP_DIVERGING = LinearSegmentedColormap.from_list('paper_div',
    [matplotlib.colors.to_rgb(c) for c in _div], N=256)

# Default heatmap aliases
CMAP_HEATMAP = CMAP_HM2
CMAP_RESIDUAL = CMAP_DIVERGING
CMAP_PAPER = CMAP_HM2

# === Functional roles ===
PALETTE = [PRIMARY, NAVY, OLIVE, HM1[3], HM2[3], SLATE]

C_OURS = PRIMARY
C_BASELINE = NAVY         # for ours-vs-baseline comparisons
C_THIRD = OLIVE

# 3-agent colors
C_AGENT = [PRIMARY, NAVY, OLIVE]

# Signpost / gradient chain
C_SIGNPOST = '#E6A817'    # warm gold diamond
C_2ND_DEGREE = HM1[3]     # #B25F79

# Per-seed detail
C_RAW = '#C5CAD0'
C_SMOOTHED = PRIMARY
C_BENCHMARK = SLATE
C_DISC = PRIMARY
C_IDENTITY = SLATE
C_AGENT0 = PRIMARY
C_OPP1 = NAVY
C_OPP2 = OLIVE

# Backward-compat aliases used in summaries/teaser
GOLD = NAVY               # baseline color
GOLD_LIGHT = NAVY_LIGHT
TEAL = OLIVE
TEAL_LIGHT = OLIVE_LIGHT
ROSE = HM2[4]
SIENNA = HM1[4]
SECONDARY = NAVY
TERTIARY = OLIVE
ACCENT = HM1[3]

EVOLUTION_CMAP = 'magma'


def apply_style():
    matplotlib.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 250,
        'font.family': 'sans-serif',
        'axes.prop_cycle': matplotlib.cycler(color=PALETTE),
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
