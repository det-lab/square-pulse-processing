#!/usr/bin/env python3
import numpy as np
import pandas as pd


# ================================================================
#  Generate Poisson-distributed pulse start times
# ================================================================
def generate_pulse_times(mean_interval=0.01, total_time=0.1, seed=12345):
    """
    Generate a list of pulse times using a Poisson process.

    mean_interval : mean time between pulses (seconds)
    total_time    : total simulated time (seconds)
    seed          : RNG seed for reproducibility
    """

    rng = np.random.default_rng(seed)

    pulse_times = []
    t = 0.0

    while t < total_time:
        # Draw waiting time from exponential distribution
        dt = rng.exponential(mean_interval)
        t += dt
        if t < total_time:
            pulse_times.append(t)

    return np.array(pulse_times)


# ================================================================
#  Add pulses to the noise signal
# ================================================================
def add_square_pulses(times, voltages, pulse_times,
                      amplitude=2.0,
                      rise_time=2e-9,
                      width=10e-9):

    total_pulse_duration = 2 * rise_time + width

    for t0 in pulse_times:

        start = t0
        end = t0 + total_pulse_duration

        mask = (times >= start) & (times < end)
        if not mask.any():
            continue

        # Index array of where the mask applies
        idx = np.where(mask)[0]

        t_relative = times[idx] - start

        # --- Rising edge
        rising_mask = (t_relative < rise_time)
        voltages[idx[rising_mask]] += amplitude * (
            t_relative[rising_mask] / rise_time
        )

        # --- Flat top
        flat_mask = (t_relative >= rise_time) & (t_relative < rise_time + width)
        voltages[idx[flat_mask]] += amplitude

        # --- Falling edge
        falling_mask = (t_relative >= rise_time + width)
        t_fall = t_relative[falling_mask] - (rise_time + width)
        voltages[idx[falling_mask]] += amplitude * (
            1 - (t_fall / rise_time)
        )

    return voltages



# ================================================================
#  Generate oscilloscope-like data
# ================================================================
def generate_scope_data(
    total_time=0.1,
    sample_rate=1e9,      # 1 sample = 1 ns
    noise_std=0.05,
    rng_seed=999,
    pulse_times=None,
    pulse_amplitude=2.0,
    rise_time=2e-9,
    width=10e-9,
):
    """
    Generate random oscilloscope-like data with noise and pulses.
    """

    rng = np.random.default_rng(rng_seed)

    num_samples = int(total_time * sample_rate)

    # Time axis
    times = np.arange(num_samples) / sample_rate

    # Noise around zero
    voltages = rng.normal(0.0, noise_std, num_samples)

    # Add pulses if provided
    if pulse_times is not None:
        voltages = add_square_pulses(
            times, voltages, pulse_times,
            amplitude=pulse_amplitude,
            rise_time=rise_time,
            width=width
        )

    return times, voltages


# ================================================================
#  Save to CSV
# ================================================================
def save_to_csv(filename, times, voltages):
    df = pd.DataFrame({"time_s": times, "voltage_V": voltages})
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


# ================================================================
#  Example usage
# ================================================================
# Simulation settings
TOTAL_TIME = 0.05     # 50 ms simulated
MEAN_INTERVAL = 0.001   # average 10 ms between pulses
SEED = 42

# Generate pulse times
pulse_times = generate_pulse_times(
    mean_interval=MEAN_INTERVAL,
    total_time=TOTAL_TIME,
    seed=SEED
)

# Generate waveform
times, volts = generate_scope_data(
    total_time=TOTAL_TIME,
    sample_rate=1e9,
    noise_std=0.05,
    rng_seed=1337,
    pulse_times=pulse_times,
    pulse_amplitude=2.0,
    rise_time=2e-9,
    width=10e-9,
)

# Save
save_to_csv("scope_data.csv", times, volts)

