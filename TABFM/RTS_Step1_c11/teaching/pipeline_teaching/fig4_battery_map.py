#!/usr/bin/env python3
"""Figure 4 — The Battery Map (icon-driven redesign)

Message: The full schedule, and where we are now.
Clean Gantt chart with gate, color-coded bars, 'You are here' cursor, and verdict star.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icons import *

setup_font()
OUT = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor(OFFWHT); ax.set_facecolor(OFFWHT)

# ---- Title ----
ax.text(8, 9.5, "Figure 4.  The Battery Map", ha='center', fontsize=16, fontweight='bold')
ax.text(8, 9.1, "The full schedule, and where we are now", ha='center', fontsize=11, color=GRAY, style='italic')

# ---- Gantt chart geometry ----
# x mapping: x = X0 + hours * SCALE
X0 = 3.2          # where hour 0 starts
SCALE = 0.095     # data units per hour
TOTAL_H = 120     # total axis range

# Row geometry
row_top = 8.2
row_spacing = 0.62
bar_h = 0.38

# Component data: (label, start_h, dur_h, color, fc, style)
components = [
    ("Gate",       0.0,   0.15, BLACK,  YELLOW_LT, 'solid'),    # prerequisite
    ("Test 1",     0.15, 16.0,  BLUE,   BLUE_LT,   'solid'),
    ("Classifier", 16.15, 6.0,  GREEN,  GREEN_LT,  'solid'),
    ("Test 3",     22.15, 0.3,  GRAY,   GRAY_LT,   'solid'),
    ("Test 2",     22.45,16.0,  BLUE,   BLUE_LT,   'solid'),
    ("Binary",     38.45,19.0,  GREEN,  GREEN_LT,  'solid'),
    ("Group Imp.", 57.45,25.0,  GREEN,  GREEN_LT,  'solid'),
    ("Width",      82.45, 0.3,  GRAY,   GRAY_LT,   'solid'),
    ("Fat Rerun",  82.75,16.0,  ORANGE, ORANGE_LT, 'solid'),
    ("Tandem",     98.75,14.0,  ORANGE, ORANGE_LT, 'solid'),
    ("Cond.",     112.75, 7.0,  ORANGE, ORANGE_LT, 'dashed'),   # conditional
]

# ---- Draw gate icon + checkmark at top left ----
gate_y = row_top
draw_gate(ax, 1.5, gate_y, 1.2, color=BLACK, fc=YELLOW_LT)
draw_check(ax, 1.5, gate_y, 0.8, color=GREEN)
ax.text(1.5, gate_y - 0.7, "+0.376", ha='center', fontsize=9, fontweight='bold', color=GREEN)

# ---- Draw bars ----
for i, (label, start, dur, color, fc, style) in enumerate(components):
    y = row_top - i * row_spacing
    x_start = X0 + start * SCALE
    x_dur = dur * SCALE
    if x_dur < 0.03:
        x_dur = 0.03  # minimum visible width for slivers

    if style == 'dashed':
        bar = Rectangle((x_start, y - bar_h/2), x_dur, bar_h,
                        facecolor=fc, edgecolor=color, lw=1.5,
                        linestyle='--', zorder=4)
    else:
        bar = Rectangle((x_start, y - bar_h/2), x_dur, bar_h,
                        facecolor=fc, edgecolor=color, lw=1.5, zorder=4)
    ax.add_patch(bar)

    # Label on left side (skip Gate — icon is self-explanatory; skip slivers — explained in caption)
    if label not in ("Gate", "Test 3", "Width"):
        ax.text(X0 - 0.15, y, label, ha='right', va='center', fontsize=8, color=BLACK)

# ---- "You are here" cursor (red vertical dashed line) ----
# Currently in Test 1, approximately 10 hours in
cursor_h = 10.0
cursor_x = X0 + cursor_h * SCALE
ax.plot([cursor_x, cursor_x], [row_top - 11 * row_spacing - 0.3, row_top + 0.5],
        color=RED, lw=2, ls='--', zorder=5)
ax.text(cursor_x, row_top + 0.7, "You are here", ha='center', fontsize=8,
        fontweight='bold', color=RED)

# ---- Verdict star at far right ----
verdict_x = X0 + TOTAL_H * SCALE + 0.5
verdict_y = row_top - 5 * row_spacing  # middle of chart
draw_star(ax, verdict_x, verdict_y, 1.2, color=BLACK)
ax.text(verdict_x, verdict_y - 0.8, "Verdict", ha='center', fontsize=9,
        fontweight='bold', color=BLACK)

# ---- Time axis at bottom ----
axis_y = row_top - 11 * row_spacing - 0.6
ax.plot([X0, X0 + TOTAL_H * SCALE], [axis_y, axis_y], color=BLACK, lw=1, zorder=3)
for h in [0, 60, 120]:
    tx = X0 + h * SCALE
    ax.plot([tx, tx], [axis_y, axis_y - 0.1], color=BLACK, lw=1, zorder=3)
    ax.text(tx, axis_y - 0.35, f"{h}h", ha='center', fontsize=7, color=GRAY)
# Intermediate tick marks (no labels)
for h in [20, 40, 80, 100]:
    tx = X0 + h * SCALE
    ax.plot([tx, tx], [axis_y, axis_y - 0.08], color=GRAY, lw=0.8, zorder=3)

# (Legend removed — color meanings explained in caption)

# ---- Caption ----
caption = ("What this shows: The full analysis schedule from start to finish. Each bar is one component; bar length is proportional to runtime.\n"
           "Blue = main tests, Green = classifier tests, Orange = reruns, Gray = quick checks. The red line marks current progress (in Test 1).\n\n"
           "What this does not show: Any results or outcomes from completed components.")
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9,
         color='#555', style='italic', linespacing=1.4)

plt.tight_layout(); plt.subplots_adjust(bottom=0.12, top=0.95)
fig.savefig(os.path.join(OUT, 'fig4_battery_map.png'), dpi=150, bbox_inches='tight', facecolor=OFFWHT)
fig.savefig(os.path.join(OUT, 'fig4_battery_map.svg'), format='svg', bbox_inches='tight', facecolor=OFFWHT)
plt.close(fig); print("Figure 4 saved.")
