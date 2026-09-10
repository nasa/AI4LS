#!/usr/bin/env python3
"""Figure 1 — The Machine Picture (icon-driven redesign)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

# Title
ax.text(7, 9.5, "Figure 1.  The Machine Picture", ha='center', fontsize=16, fontweight='bold')
ax.text(7, 9.1, "Your work survives any computer failure", ha='center', fontsize=11, color=GRAY, style='italic')

# Laptop (small, top)
draw_laptop(ax, 7, 8.0, 1.2)
ax.text(7, 7.2, "Your Laptop", ha='center', fontsize=9, color=GRAY)

# Arrow down
arrow(7, 7.0, 7, 6.5, c=GRAY, lw=1.5)

# Gear (medium, center)
draw_gear(ax, 7, 6.0, 1.3)
ax.text(7, 5.2, "Coordinator", ha='center', fontsize=9, color=GRAY)

# Arrows splitting to server and vault
arrow(6.3, 5.7, 4.0, 4.5, c=BLACK, lw=2)
arrow(7.7, 5.7, 10.0, 4.5, c=BLACK, lw=2)

# Server (large, left) — the engine
draw_server(ax, 3.5, 3.5, 2.8)
ax.text(3.5, 2.2, "GPU Computer", ha='center', fontsize=10, fontweight='bold', color=BLUE)

# Vault (large, right) — the vault
draw_vault(ax, 10.5, 3.5, 2.8)
ax.text(10.5, 2.2, "Permanent Storage", ha='center', fontsize=10, fontweight='bold', color=GREEN)

# Double arrow between server and vault
arrow(5.2, 3.5, 8.8, 3.5, c=BLACK, lw=2, s='<->')

# Warning icon on server (crash)
draw_warning(ax, 3.5, 5.2, 1.0)
# Curved arrow from warning to vault (saves)
arrow(4.2, 5.2, 9.5, 4.2, c=ORANGE, lw=1.5, cs='arc3,rad=-0.3')
ax.text(7.0, 5.5, "saves", ha='center', fontsize=8, color=ORANGE, style='italic')

# Bottom crash recovery strip
strip_y = 1.2
# Warning → Clock → Check
draw_warning(ax, 3.0, strip_y, 1.0)
arrow(3.6, strip_y, 6.3, strip_y, c=GRAY, lw=1.5)
draw_clock(ax, 7.0, strip_y, 1.0)
arrow(7.6, strip_y, 10.3, strip_y, c=GRAY, lw=1.5)
draw_check(ax, 11.0, strip_y, 1.0, color=GREEN)

ax.text(3.0, 0.4, "If crash", ha='center', fontsize=8, color=ORANGE)
ax.text(7.0, 0.4, "~1.5h lost", ha='center', fontsize=8, color=GRAY)
ax.text(11.0, 0.4, "Resume", ha='center', fontsize=8, color=GREEN)

# Caption
caption = ("What this shows: The analysis runs on a GPU computer, not your laptop.\n"
           "Everything that matters lives on permanent storage that survives any crash.\n"
           "If the computer dies, at most ~1.5 hours of work is lost.\n\n"
           "What this does not show: Network details or any scientific results.")
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9,
         color='#555', style='italic', linespacing=1.4)

plt.tight_layout(); plt.subplots_adjust(bottom=0.12, top=0.95)
fig.savefig(os.path.join(OUT, 'fig1_machine_picture.png'), dpi=150, bbox_inches='tight', facecolor=OFFWHT)
fig.savefig(os.path.join(OUT, 'fig1_machine_picture.svg'), format='svg', bbox_inches='tight', facecolor=OFFWHT)
plt.close(fig); print("Figure 1 saved.")
