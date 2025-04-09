import matplotlib
matplotlib.use('TkAgg')  # <- Add this first

from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt

# Create a vertical half pitch (goal at the bottom)
pitch = VerticalPitch(
    half=True,
    pitch_color='darkgreen',
    line_color='white',
    linewidth=2,
    goal_type='box',
    label=True,
    axis=True
)

fig, ax = pitch.draw(figsize=(6, 10))  # Taller figure for vertical layout
ax.invert_yaxis()

# Plot a transformed shot
pitch.scatter([63.0], [109.45], ax=ax, color='red', marker='o', s=300, zorder=5, label='Transformed Shot')

ax.legend()
plt.title("Vertical Half Pitch with Goal at Bottom")

# Optional: print clicked coordinates
def onclick(event):
    if event.inaxes == ax:
        print(f"Mouse clicked at: ({event.xdata:.2f}, {event.ydata:.2f})")

fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
