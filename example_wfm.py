import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from analyze_single_photon_data import count_pulses_in_interval
import numpy as np
import RigolWFM.wfm as rigol

# make the plots publication-ready
plt.style.use('./mplstyles/standard.mplstyle')

# point to your file
wfm_filename = "data/NewFile2.wfm" 

w = rigol.Wfm.from_file(wfm_filename, '1000Z')

description = w.describe()
print(description)

ch = w.channels[0]
print(ch.times)


# compute timestep
dt = ch.times[1] - ch.times[0]

# check timestep
if not np.isclose(dt, 1e-9):
    raise ValueError(f"Unexpected timestep: {dt} s. Expected 1e-9 s.")

# build a dataframe that has the right columns for
# the function count_pulses_in_interval
df = pd.DataFrame({
    "time_s": ch.times,
    "voltage_V": ch.volts  # assuming the channel voltage values are in ch.data
})

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

# Step plot for histogram fill (outer outline)
x_step_hist = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
y_step_hist = np.concatenate([hist_counts, [0]])  # drop to zero at the end

# Step plot for Poisson PMF
x_step_pmf = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
y_step_pmf = np.concatenate([pmf_int, [pmf_int[-1]]])

# -----------------------------
# Plot everything
# -----------------------------
plt.figure(figsize=(10, 6))

# Histogram as filled step (light blue fill, thin dark blue outline)
plt.fill_between(
    x_step_hist, 0, y_step_hist,
    step='post',
    color='#add8e6',          # light blue fill
    edgecolor='#1f4e79',      # dark blue outline
    linewidth=1.5,
    label='Counts per interval'
)

# Error bars (mid-dark blue, thicker)
plt.errorbar(
    x_centers, hist_counts, yerr=errors, fmt='none',
    ecolor='#1f4e79', elinewidth=3, capsize=6, capthick=3,
    label='√N error bars'
)

# Poisson PMF step plot (black, thicker)
plt.step(
    x_step_pmf, y_step_pmf, where='post',
    color='black', linewidth=3,
    label=f'Poisson PMF (λ={lambda_est:.2f})'
)

# -----------------------------
# Adjust y limits
# -----------------------------
ymax = int(np.ceil((hist_counts + errors).max()))
plt.ylim(0, ymax)

# Labels, grid, legend
plt.xlabel("Pulses per interval")
plt.ylabel("Counts")
plt.xticks(bin_edges)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()