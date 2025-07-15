from pathlib import Path

import numpy as np
import partipy as pt
import matplotlib.pyplot as plt

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "delta_visualization"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "delta_visualization"
output_dir.mkdir(exist_ok=True, parents=True)

N_SAMPLES = 2_000
N_DIMENSION = 2
N_ARCHETYPES = 3
NOISE_STD = 0.10
SEED = 153

X, A, Z = pt.simulate_archetypes(n_samples=N_SAMPLES, n_archetypes=N_ARCHETYPES, n_dimensions=N_DIMENSION, noise_std=NOISE_STD, seed=SEED)
X = X.astype(np.float32)

# Delta values to test
delta_values = [0.0, 0.1, 0.2, 0.4]

# Create figure with 1 row and 5 columns
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i, delta in enumerate(delta_values):
    ax = axes[i]
    
    # Fit archetypal analysis with current delta
    AA_object = pt.AA(n_archetypes=3, delta=delta)
    AA_object.fit(X)
    Z_hat = AA_object.Z
    
    # Plot on current subplot
    ax.grid(alpha=0.5)
    ax.scatter(x=X[:, 0], y=X[:, 1], s=3, c="blue", label="Data Point")
    ax.scatter(x=Z[:, 0], y=Z[:, 1], s=20, c="red", label="True\nArchetype")
    ax.scatter(x=Z_hat[:, 0], y=Z_hat[:, 1], s=20, c="green", label="Inferred\nArchetype")
    
    # Draw the convex hull
    Z_loop = np.vstack([Z_hat, Z_hat[0]])
    ax.plot(Z_loop[:, 0], Z_loop[:, 1], c="green", linestyle='-', linewidth=1)
    
    # Set title and formatting
    ax.set_title(f"delta={delta:.1f} | RSS={AA_object.RSS:.3f}")
    ax.legend()
    ax.axis("equal")

plt.tight_layout()

fig.savefig(figure_dir / "delta_comparison.pdf", bbox_inches="tight")
fig.savefig(figure_dir / "delta_comparison.jpg", bbox_inches="tight", dpi=300)