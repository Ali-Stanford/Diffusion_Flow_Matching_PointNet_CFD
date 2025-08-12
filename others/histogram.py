import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Path to your relative-error data
file_path = 'u_collection.txt'

# Set x-axis limits (uncomment as needed)
u_max = 0.18
v_max = 0.55
p_max = 0.27

# Load the 222 relative-error values from the text file
errors = np.loadtxt(file_path)

# Fixed bin width (thickness)
bin_width = 0.0035  # change this value as desired

# Define bin edges from 0.0 to 0.27 with fixed step
#bins = np.arange(0.0, 0.27 + bin_width, bin_width)
bins = np.arange(0.0, u_max + bin_width, bin_width)

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with fixed-width bins
ax.hist(errors, bins=bins, edgecolor='black')

# Set x-axis limits
ax.set_xlim(0.00, u_max)  # for u
#ax.set_xlim(0.00, v_max)  # for v
#ax.set_xlim(0.00, p_max)  # for p

# Force integer y-ticks
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Move tick labels away from the axis line
ax.tick_params(axis='x', which='major', pad=8)
ax.tick_params(axis='y', which='major', pad=8)

# Set axis labels
ax.set_xlabel(
    r'Relative error in $\mathit{L}^2$ norm',
    fontsize=15 * 1.8,
    labelpad=15
)
ax.set_ylabel(
    'Frequency',
    fontsize=15 * 1.8,
    labelpad=15
)

# Title
#ax.set_title('Prediction of gauge pressure', fontsize=15*1.8)
ax.set_title('Prediction of velocity $\\mathit{u}$', fontsize=15 * 1.8)

# X-axis tick spacing
ax.xaxis.set_major_locator(MultipleLocator(0.02))

# Tick-label font sizes
ax.tick_params(axis='x', labelsize=14 * 1.3)
ax.tick_params(axis='y', labelsize=14 * 1.3)

# Tight layout and save
plt.tight_layout()
plt.savefig('u_collection.pdf')
plt.show()

