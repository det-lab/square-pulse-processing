import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from analyze_single_photon_data import count_pulses_in_interval
import numpy as np

df = pd.read_csv("scope_data.csv")

interval_results = count_pulses_in_interval(
    df,
    interval_length=0.001,   # 1 ms bins
    threshold=1.0            # half the 2 V pulse amplitude
)

pulse_counts = interval_results["pulse_count"].values

# -----------------------------
# Histogram settings
# -----------------------------
max_count = pulse_counts.max()
bin_edges = np.arange(0, max_count + 2)  # left edges of integer bins
bin_width = 1

hist_counts, _ = np.histogram(pulse_counts, bins=bin_edges)
errors = np.sqrt(hist_counts)  # Poisson uncertainty

# Bin centers for error bars
x_centers = bin_edges[:-1] + 0.5

# -----------------------------
# Poisson PMF
# -----------------------------
lambda_est = pulse_counts.mean()
x_int = np.arange(0, max_count + 1)
pmf_int = poisson.pmf(x_int, mu=lambda_est) * len(interval_results)

# Step plot aligned with histogram bars
# Use 'where=post', append one extra x (right edge of last bin) and last y-value
x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
y_step = np.concatenate([pmf_int, [pmf_int[-1]]])

# -----------------------------
# Plot everything
# -----------------------------
plt.figure(figsize=(10, 6))

# Histogram bars (light blue, left-aligned)
plt.bar(bin_edges[:-1], hist_counts, width=bin_width, align='edge',
        color='#add8e6', edgecolor='black', label='Counts per interval')

# Error bars (mid-dark blue, thicker)
plt.errorbar(x_centers, hist_counts, yerr=errors, fmt='none',
             ecolor='#1f4e79', elinewidth=2, capsize=4, label='√N error bars')

# Poisson PMF step plot (black, thicker)
plt.step(x_step, y_step, where='post', color='black', linewidth=3,
         label=f'Poisson PMF (λ={lambda_est:.2f})')

# Labels, grid, legend
plt.xlabel("Pulses per interval")
plt.ylabel("Number of intervals")
plt.title("Histogram of Pulse Counts with Poisson PMF")
plt.xticks(bin_edges)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()