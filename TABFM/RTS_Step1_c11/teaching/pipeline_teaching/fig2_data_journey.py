#!/usr/bin/env python3
"""Figure 2 — The Data Journey (icon-driven redesign)

Message: Only safe data reaches the model.
Left-to-right: archive -> big grid -> funnel (safety filter) -> small grid -> chip
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icons import *

setup_font()
OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor(OFFWHT); ax.set_facecolor(OFFWHT)

def arrow(x1, y1, x2, y2, c=BLACK, lw=2, s='->', cs='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=s,
                 color=c, linewidth=lw, connectionstyle=cs, mutation_scale=16, zorder=3))

# ---- Inline helper: pill icon (treatment) ----
def draw_pill(ax, x, y, s, color=ORANGE, fc=ORANGE_LT):
    w, h = s*0.35, s*0.15
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.02',
                          facecolor=fc, edgecolor=color, lw=2, zorder=4)
    ax.add_patch(rect)
    ax.plot([x, x], [y-h/2+0.02, y+h/2-0.02], color=color, lw=1.5, zorder=5)

# ---- Title ----
ax.text(8, 9.5, "Figure 2.  The Data Journey", ha='center', fontsize=16, fontweight='bold')
ax.text(8, 9.1, "Only safe data reaches the model", ha='center', fontsize=11, color=GRAY, style='italic')

# ---- Main flow (left to right at y=5) ----
flow_y = 4.8

# Archive (left)
draw_archive(ax, 2.0, flow_y, 1.8)
ax.text(2.0, 3.5, "NASA Archive", ha='center', fontsize=9, color=GRAY)

# Arrow to big grid
arrow(2.9, flow_y, 3.8, flow_y, c=GRAY, lw=1.5)

# Big grid (43 x 7,545) — wide
draw_grid(ax, 5.2, flow_y, w=2.2, h=3.0, color=BLACK, fc=GRAY_LT, rows=6, cols=10)
ax.text(5.2, 3.2, "43 x 7,545", ha='center', fontsize=9, fontweight='bold', color=BLACK)

# Arrow into funnel
arrow(6.5, flow_y, 7.3, flow_y, c=GRAY, lw=1.5)

# Funnel (center, large — visual anchor)
draw_funnel(ax, 8.3, flow_y, 3.0, color=ORANGE, fc=ORANGE_LT)
ax.text(8.3, 3.2, "Safety Filter", ha='center', fontsize=10, fontweight='bold', color=ORANGE)

# ---- 3 "removed" icons above funnel ----
removed_y = 7.8
# Subject ID (badge + X)
draw_badge(ax, 6.8, removed_y, 1.2, color=BLACK, fc=GRAY_LT)
draw_x(ax, 7.5, removed_y + 0.3, 0.8, color=RED)
ax.text(6.8, removed_y - 0.7, "subject ID", ha='center', fontsize=8, color=GRAY)

# Future data (calendar + X)
draw_calendar(ax, 8.3, removed_y, 1.2, color=ORANGE, fc=OFFWHT)
draw_x(ax, 9.0, removed_y + 0.3, 0.8, color=RED)
ax.text(8.3, removed_y - 0.7, "future data", ha='center', fontsize=8, color=GRAY)

# Treatment (pill + X)
draw_pill(ax, 9.8, removed_y, 1.2, color=ORANGE, fc=ORANGE_LT)
draw_x(ax, 10.5, removed_y + 0.3, 0.8, color=RED)
ax.text(9.8, removed_y - 0.7, "treatment", ha='center', fontsize=8, color=GRAY)

# Small downward arrows from removed icons to funnel top
arrow(6.8, removed_y - 0.3, 7.8, flow_y + 1.0, c=RED, lw=1, cs='arc3,rad=0.15')
arrow(8.3, removed_y - 0.3, 8.3, flow_y + 1.0, c=RED, lw=1)
arrow(9.8, removed_y - 0.3, 8.8, flow_y + 1.0, c=RED, lw=1, cs='arc3,rad=-0.15')

# Arrow from funnel to small grid
arrow(8.9, flow_y - 0.6, 10.5, flow_y, c=GRAY, lw=1.5)

# Small grid (38 x 628) — narrow
draw_grid(ax, 11.5, flow_y, w=1.4, h=2.0, color=GREEN, fc=GREEN_LT, rows=5, cols=6)
ax.text(11.5, 3.2, "38 x 628", ha='center', fontsize=9, fontweight='bold', color=GREEN)

# Arrow to chip
arrow(12.4, flow_y, 13.2, flow_y, c=GRAY, lw=1.5)

# Chip/model (right)
draw_chip(ax, 14.0, flow_y, 1.8, color=BLUE, fc=BLUE_LT)
ax.text(14.0, 3.5, "Model", ha='center', fontsize=10, fontweight='bold', color=BLUE)

# ---- Caption ----
caption = ("What this shows: Raw NASA data (43 subjects, 7,545 measurements) passes through a safety filter.\n"
           "Subject IDs, future data, and treatment labels are removed. Only 38 subjects x 628 safe features reach the model.\n\n"
           "What this does not show: The specific filtering rules or any model results.")
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9,
         color='#555', style='italic', linespacing=1.4)

plt.tight_layout(); plt.subplots_adjust(bottom=0.10, top=0.95)
fig.savefig(os.path.join(OUT, 'fig2_data_journey.png'), dpi=150, bbox_inches='tight', facecolor=OFFWHT)
fig.savefig(os.path.join(OUT, 'fig2_data_journey.svg'), format='svg', bbox_inches='tight', facecolor=OFFWHT)
plt.close(fig); print("Figure 2 saved.")
