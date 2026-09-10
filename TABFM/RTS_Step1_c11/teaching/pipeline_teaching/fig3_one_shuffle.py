#!/usr/bin/env python3
"""Figure 3 — One Shuffle (icon-driven redesign)

Message: One test = 500 shuffles = 16 hours. Here's why.
Central loop: Mix -> Run model -> Score -> Add to pile, repeated 500 times.
Visual payoff: histogram showing real result vs. cloud of chance.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Circle
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icons import *

setup_font()
OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor(OFFWHT); ax.set_facecolor(OFFWHT)

def arrow(x1, y1, x2, y2, c=BLACK, lw=2, s='->', cs='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=s,
                 color=c, linewidth=lw, connectionstyle=cs, mutation_scale=16, zorder=3))

# ---- Title ----
ax.text(7, 9.5, "Figure 3.  One Shuffle", ha='center', fontsize=16, fontweight='bold')
ax.text(7, 9.1, "One test = 500 shuffles = 16 hours", ha='center', fontsize=11, color=GRAY, style='italic')

# ---- Top: Real Result star ----
draw_star(ax, 7, 8.3, 1.0, color=BLACK)
ax.text(7, 7.7, "Real Result", ha='center', fontsize=9, color=GRAY)

# ---- Central loop ----
loop_cx, loop_cy = 7, 4.5
loop_r = 2.3

# Draw the circular path (dashed)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(loop_cx + loop_r * np.cos(theta), loop_cy + loop_r * np.sin(theta),
        color=ORANGE, lw=2, ls='--', alpha=0.4, zorder=2)

# 4 icons at cardinal points
# Top: Cards (Mix)
draw_cards(ax, loop_cx, loop_cy + loop_r, 1.5, color=ORANGE, fc=ORANGE_LT)
ax.text(loop_cx, loop_cy + loop_r - 0.7, "Mix", ha='center', fontsize=9, color=GRAY)

# Right: Server + 38x badge (Run model)
draw_server(ax, loop_cx + loop_r, loop_cy, 1.5, color=BLUE, fc=BLUE_LT)
ax.text(loop_cx + loop_r, loop_cy - 0.9, "Run model", ha='center', fontsize=9, color=GRAY)
# 38x badge
ax.text(loop_cx + loop_r + 0.5, loop_cy + 0.5, "38x", ha='center', fontsize=8,
        fontweight='bold', color=BLUE,
        bbox=dict(boxstyle='round,pad=0.15', facecolor=BLUE_LT, edgecolor=BLUE, lw=1))

# Bottom: Check (Score)
draw_check(ax, loop_cx, loop_cy - loop_r, 1.3, color=GREEN)
ax.text(loop_cx, loop_cy - loop_r - 0.5, "Score", ha='center', fontsize=9, color=GRAY)

# Left: Bucket (Add to pile)
draw_bucket(ax, loop_cx - loop_r, loop_cy, 1.5, color=GRAY, fc=GRAY_LT)
ax.text(loop_cx - loop_r, loop_cy - 0.9, "Add to pile", ha='center', fontsize=9, color=GRAY)

# Center: x500 + loop arrow
ax.text(loop_cx, loop_cy + 0.2, "x500", ha='center', va='center', fontsize=22,
        fontweight='bold', color=ORANGE)
draw_loop(ax, loop_cx, loop_cy - 0.8, 1.5, color=ORANGE)

# Curved arrows along the loop (clockwise)
for start_ang, end_ang in [(90, 0), (0, -90), (-90, 180), (180, 90)]:
    t = np.linspace(np.radians(start_ang), np.radians(end_ang), 30)
    sx = loop_cx + loop_r * np.cos(t[5])
    sy = loop_cy + loop_r * np.sin(t[5])
    ex = loop_cx + loop_r * np.cos(t[-6])
    ey = loop_cy + loop_r * np.sin(t[-6])
    arrow(sx, sy, ex, ey, c=ORANGE, lw=1.5, cs='arc3,rad=-0.3')

# ---- Bottom left: Clock + 16 hours ----
draw_clock(ax, 2.5, 1.5, 1.5, color=BLACK, fc=OFFWHT)
ax.text(2.5, 0.5, "16 hours", ha='center', fontsize=10, fontweight='bold', color=BLACK)

# ---- Bottom right: Histogram (visual payoff) ----
hist_x, hist_y = 11.0, 1.5
hist_w, hist_h = 4.0, 2.0
# Bell-shaped chance distribution
heights = [0.08, 0.15, 0.28, 0.45, 0.62, 0.75, 0.68, 0.50, 0.32, 0.18, 0.08]
draw_histogram(ax, hist_x, hist_y, hist_w, hist_h, heights,
               color=GRAY, fc=GRAY_LT)
# Star marker for real result (far right, above bars)
star_x = hist_x + hist_w/2 - 0.15
star_y = hist_y - hist_h/2 + hist_h * 0.85
draw_star(ax, star_x, star_y, 0.9, color=RED)
# Arrow from star down to indicate it's the real result
ax.text(hist_x, hist_y - hist_h/2 - 0.4, "Real vs. Chance", ha='center', fontsize=9,
        fontweight='bold', color=BLACK)

# ---- Caption ----
caption = ("What this shows: Each test shuffles the data 500 times. Each shuffle: mix labels, run the model on 38 subjects,\n"
           "score the result, add it to the pile. After 500 shuffles (~16 hours), compare the real result to the chance pile.\n\n"
           "What this does not show: The statistical test details or any actual result values.")
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9,
         color='#555', style='italic', linespacing=1.4)

plt.tight_layout(); plt.subplots_adjust(bottom=0.12, top=0.95)
fig.savefig(os.path.join(OUT, 'fig3_one_shuffle.png'), dpi=150, bbox_inches='tight', facecolor=OFFWHT)
fig.savefig(os.path.join(OUT, 'fig3_one_shuffle.svg'), format='svg', bbox_inches='tight', facecolor=OFFWHT)
plt.close(fig); print("Figure 3 saved.")
