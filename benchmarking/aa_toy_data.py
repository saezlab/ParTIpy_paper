
import matplotlib.pyplot as plt
import partipy as pt

N_SAMPLES = 2_000
N_ARCHETYPES = 3
N_DIMENSIONS = 2

X, A_true, Z_true = pt.simulate_archetypes(n_samples=N_SAMPLES,
                                           n_archetypes=N_ARCHETYPES,
                                           n_dimensions=N_DIMENSIONS,
                                           noise_std=0.1, 
                                           seed=111)

feature_means = X.mean(axis=0, keepdims=True)
feature_stds = X.std(axis=0, keepdims=True)
X -= feature_means
X /= feature_stds

Z_true -= feature_means
Z_true /= feature_stds

A, B, Z, RSS, varexpl = (
    pt.AA(
          n_archetypes=N_ARCHETYPES,
          init="furthest_sum",
          #optim="regularized_nnls",
          #optim="projected_gradients",
          optim="frank_wolfe",
          # weight="bisquare",
          weight=None,
          )
    .fit(X=X)
    .return_all()
)

plt.style.use("dark_background")
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(Z[:, 0], Z[:, 1], color="red", label="approx")
plt.scatter(Z_true[:, 0], Z_true[:, 1], color="blue", label="true")
plt.legend()
plt.show()
