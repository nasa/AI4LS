import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch

# Sample data structure, assuming you've updated your JSON structure as shown above
experiment_data = {
    "experiment_groups": [
        {
            "group": "1-10",
            "mice": 120,
            "gender": "female",
            "strand": "B6CF1",
            "dose": 2.6,
            "sample_type": "Tissue",
            "samples_collected": 937
        },
        {
            "group": "11-20",
            "mice": 90,
            "gender": "male",
            "strand": "B6CF1",
            "dose": 5.2,
            "sample_type": "Tissue",
            "samples_collected": 850
        }
        # ... other groups
    ]
}

# Initialize figure and axes
fig, ax = plt.subplots(figsize=(10, 4 * len(experiment_data['experiment_groups'])))  # Adjust figure height as needed
ax.axis('off')  # Hide the axes

# Function to create an icon
def create_icon(ax, pos, path, zoom):
    icon = plt.imread(path)
    imagebox = OffsetImage(icon, zoom=zoom)
    ab = AnnotationBbox(imagebox, pos, frameon=False)
    ax.add_artist(ab)

# Function to draw an arrow
def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='->',
                            mutation_scale=15,
                            color='black',
                            lw=1)
    ax.add_patch(arrow)

# Define starting position for the first group
start_y = 0.9

# Define zoom for the icons
mouse_zoom = 0.05  # Reduced zoom for mouse icon to prevent overlap
radiation_zoom = 0.05  # Base zoom for radiation icon
sample_zoom = 0.05  # Zoom for sample icon

for group_data in experiment_data["experiment_groups"]:
    # Mouse icon
    mouse_icon_path = './icons/mouse_icon.png'
    mouse_pos = (0.1, start_y)
    create_icon(ax, mouse_pos, mouse_icon_path, zoom=mouse_zoom)

    # Strand name and gender
    strand_text_pos = (mouse_pos[0], mouse_pos[1] - 0.1)  # Increased offset for the strand text
    gender_strand_text = f"{group_data['gender'].capitalize()} {group_data['strand']}"
    ax.text(strand_text_pos[0], strand_text_pos[1], gender_strand_text, ha='center', va='top', fontsize=9)

    # Group number and number of mice
    text_x_offset = 0.10  # Increase offset for text to the right of the mouse icon
    group_text_pos = (mouse_pos[0] + text_x_offset, start_y)
    ax.text(group_text_pos[0], group_text_pos[1], f"Group {group_data['group']}", ha='left', va='center', fontsize=10)
    ax.text(group_text_pos[0], group_text_pos[1] - 0.05, f"{group_data['mice']} mice", ha='left', va='center', fontsize=10)

    # Radiation icon, scaled by the dose
    radiation_icon_path = './icons/radiation_icon.png'
    radiation_pos = (group_text_pos[0] + 0.15, start_y)  # Adjusted position for radiation icon
    radiation_zoom_scaled = radiation_zoom + 0.02 * group_data['dose']
    create_icon(ax, radiation_pos, radiation_icon_path, zoom=radiation_zoom_scaled)

    # Radiation level text
    radiation_text_pos = (radiation_pos[0], radiation_pos[1] - 0.08)  # Position for radiation level text, adjust as needed
    ax.text(radiation_text_pos[0], radiation_text_pos[1], f"{group_data['dose']} Gy", 
            ha='center', va='top', fontsize=9)

    # Right arrow from radiation to sample
    arrow_start = (radiation_pos[0] + radiation_zoom_scaled / 2 + 0.02, start_y)
    arrow_end = (arrow_start[0] + 0.1, start_y)
    draw_arrow(ax, arrow_start, arrow_end)

    # Sample type collected icon
    sample_icon_path = './icons/tissue_icon.png'
    sample_pos = (arrow_end[0] + 0.05, start_y)
    create_icon(ax, sample_pos, sample_icon_path, zoom=sample_zoom)

    # Sample type and number collected text
    sample_text = f"{group_data['sample_type']}:\n{group_data['samples_collected']} samples"
    ax.text(sample_pos[0] + 0.1, start_y, sample_text, ha='left', va='center', fontsize=9)

    # Update the y position for the next row
    start_y -= 0.4  # Adjust spacing between groups as needed

# Save the created figure
plt.savefig('./experimental_design.png', dpi=300)

# Display the figure
plt.show()

