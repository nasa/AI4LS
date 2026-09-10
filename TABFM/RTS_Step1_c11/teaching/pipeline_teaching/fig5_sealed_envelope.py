#!/usr/bin/env python3
"""Figure 5 — The Sealed-Envelope Protocol (icon-driven redesign)

Message: Rules locked first. Results read once. No peeking.
3 panels: Lock -> Run Blind -> Open Once. Bottom strip: no peeking, no cherry-picking, trustworthy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
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

# ---- Inline helper: document/file icon ----
def draw_document(ax, x, y, s, color=BLACK, fc=OFFWHT):
    w, h = s*0.22, s*0.28
    fold = s*0.06
    pts = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2-fold),
           (x+w/2-fold, y+h/2), (x-w/2, y+h/2)]
    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=color, lw=1.5, zorder=4))
    ax.plot([x+w/2-fold, x+w/2], [y+h/2, y+h/2-fold], color=color, lw=1, zorder=5)

# ---- Title ----
ax.text(8, 9.5, "Figure 5.  The Sealed-Envelope Protocol", ha='center', fontsize=16, fontweight='bold')
ax.text(8, 9.1, "Rules locked first. Results read once. No peeking.", ha='center', fontsize=11, color=GRAY, style='italic')

# ---- 3 panels ----
panel_y_top = 8.2
panel_y_bot = 3.5
panel_h = panel_y_top - panel_y_bot

# Panel backgrounds
panel1 = FancyBboxPatch((0.5, panel_y_bot), 4.5, panel_h, boxstyle='round,pad=0.1',
                        facecolor=YELLOW_LT, edgecolor=GRAY, lw=1, alpha=0.5, zorder=1)
panel2 = FancyBboxPatch((5.5, panel_y_bot), 5.0, panel_h, boxstyle='round,pad=0.1',
                        facecolor=BLUE_LT, edgecolor=GRAY, lw=1, alpha=0.5, zorder=1)
panel3 = FancyBboxPatch((11.0, panel_y_bot), 4.5, panel_h, boxstyle='round,pad=0.1',
                        facecolor=GREEN_LT, edgecolor=GRAY, lw=1, alpha=0.5, zorder=1)
ax.add_patch(panel1); ax.add_patch(panel2); ax.add_patch(panel3)

# (Panel numbers removed to meet text-element budget — panels are self-labeling)

# ---- Panel 1: Lock ----
p1_cx = 2.75
p1_cy = 5.8
draw_envelope(ax, p1_cx, p1_cy, 2.2, color=BLACK, fc=OFFWHT, open=False)
draw_lock(ax, p1_cx, p1_cy + 0.8, 1.3, color=BLACK, fc=GRAY_LT)
ax.text(p1_cx, panel_y_bot + 0.3, "Lock", ha='center', fontsize=11,
        fontweight='bold', color=BLACK)

# Arrow panel 1 -> panel 2
arrow(5.2, p1_cy, 5.3, p1_cy, c=BLACK, lw=2)

# ---- Panel 2: Run Blind ----
p2_cx = 8.0
p2_cy = 5.8
# Envelope sitting closed
draw_envelope(ax, p2_cx - 0.8, p2_cy, 1.8, color=BLACK, fc=OFFWHT, open=False)
draw_lock(ax, p2_cx - 0.8, p2_cy + 0.7, 0.9, color=BLACK, fc=GRAY_LT)
# Server running beside it
draw_server(ax, p2_cx + 1.2, p2_cy, 1.6, color=BLUE, fc=BLUE_LT)
# Pulse line above
draw_pulse(ax, p2_cx, p2_cy + 1.5, 3.0, 0.5, color=BLUE)
# Small locked file icons below
for fx in [p2_cx - 1.0, p2_cx, p2_cx + 1.0]:
    draw_document(ax, fx, p2_cy - 1.3, 0.8, color=GRAY, fc=GRAY_LT)
    draw_lock(ax, fx, p2_cy - 1.0, 0.5, color=GRAY, fc=GRAY_LT)
ax.text(p2_cx, panel_y_bot + 0.3, "Run Blind", ha='center', fontsize=11,
        fontweight='bold', color=BLUE)

# Arrow panel 2 -> panel 3
arrow(10.7, p2_cy, 10.8, p2_cy, c=BLACK, lw=2)

# ---- Panel 3: Open Once ----
p3_cx = 13.25
p3_cy = 5.8
draw_envelope(ax, p3_cx, p3_cy, 2.2, color=BLACK, fc=OFFWHT, open=True)
draw_unlock(ax, p3_cx, p3_cy + 0.8, 1.3, color=GREEN, fc=GREEN_LT)
ax.text(p3_cx, panel_y_bot + 0.3, "Open Once", ha='center', fontsize=11,
        fontweight='bold', color=GREEN)

# ---- Verdict star (far right, after panel 3) ----
draw_star(ax, 15.5, p3_cy, 1.0, color=BLACK)
ax.text(15.5, p3_cy - 0.7, "Verdict", ha='center', fontsize=9,
        fontweight='bold', color=BLACK)

# ---- Bottom strip (orange): no peeking, no cherry-picking, trustworthy ----
strip_y = 2.0
strip = FancyBboxPatch((1.0, 0.8), 14.0, 1.8, boxstyle='round,pad=0.1',
                       facecolor=ORANGE_LT, edgecolor=ORANGE, lw=1.5, alpha=0.4, zorder=1)
ax.add_patch(strip)

# No peeking (eye + X)
draw_eye(ax, 3.5, strip_y, 1.3, color=BLACK, fc=OFFWHT)
draw_x(ax, 4.1, strip_y + 0.3, 0.8, color=RED)
ax.text(3.5, strip_y - 0.7, "No peeking", ha='center', fontsize=9, color=BLACK)

# No cherry-picking (cards + X)
draw_cards(ax, 8.0, strip_y, 1.3, color=ORANGE, fc=ORANGE_LT)
draw_x(ax, 8.6, strip_y + 0.3, 0.8, color=RED)
ax.text(8.0, strip_y - 0.7, "No cherry-picking", ha='center', fontsize=9, color=BLACK)

# Trustworthy (lock + check)
draw_lock(ax, 12.5, strip_y, 1.3, color=GREEN, fc=GREEN_LT)
draw_check(ax, 12.5, strip_y, 0.9, color=GREEN)
ax.text(12.5, strip_y - 0.7, "Trustworthy", ha='center', fontsize=9, color=BLACK)

# ---- Caption ----
caption = ("What this shows: Analysis rules are locked before any test runs (Step 1). All tests run with results sealed\n"
           "away — nobody sees them until every test finishes (Step 2). Only then is the envelope opened once (Step 3).\n"
           "No peeking at intermediate results. No cherry-picking the best outcome.\n\n"
           "What this does not show: The specific rules locked or the final verdict.")
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9,
         color='#555', style='italic', linespacing=1.4)

plt.tight_layout(); plt.subplots_adjust(bottom=0.14, top=0.95)
fig.savefig(os.path.join(OUT, 'fig5_sealed_envelope.png'), dpi=150, bbox_inches='tight', facecolor=OFFWHT)
fig.savefig(os.path.join(OUT, 'fig5_sealed_envelope.svg'), format='svg', bbox_inches='tight', facecolor=OFFWHT)
plt.close(fig); print("Figure 5 saved.")
