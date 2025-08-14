import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_theta_u(path):
    """Load first two columns: theta (rad) and u; return theta in degrees and u."""
    data = np.loadtxt(path, usecols=(0, 1))
    theta_rad, u = data[:, 0], data[:, 1]
    theta_deg = np.degrees(theta_rad)
    # (Optional) sort by theta for cleaner lines if file is unsorted
    order = np.argsort(theta_deg)
    return theta_deg[order], u[order]

# File specs: (filename, label, linestyle, color)
series = [
    ("v_gt.txt",        "Ground truth", "-",  'black'),
    ("v_flow.txt",      "Flow Matching PointNet",         "--", "blue"),
    ("v_diffusion.txt", "Diffusion PointNet",    "-.", "orange"),
    ("v_pure.txt",      "PointNet",         ":",  "C3"),
]

fig, ax = plt.subplots(figsize=(7, 4.5))

for fname, label, ls, color in series:
    path = Path(fname)
    if not path.exists():
        print(f"Warning: {fname} not found; skipping.")
        continue
    theta_deg, u = load_theta_u(path)
    ax.plot(theta_deg, u, linestyle=ls, linewidth=2, color=color, label=label)

ax.set_xlabel(r"$\theta ^\circ$",fontsize=20)
#ax.set_ylabel("Gauge pressure (Pa)",fontsize=20)
ax.set_ylabel(r"$\mathit{v}$ (m/s)",fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
#ax.legend()
#ax.legend(fontsize=18)
plt.tight_layout()
#plt.show()
# To save instead of (or in addition to) showing:
plt.savefig("v_vs_theta.png", dpi=300)
plt.savefig("v_vs_theta.pdf")
