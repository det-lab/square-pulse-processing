import numpy as np
import pandas as pd


def generate_pulse_times(duration_s, mean_time_between_pulses_s=0.01):
    """
    Generate pulse trigger times (in sample indices = nanoseconds)
    using exponential waiting times (Poisson process).
    """

    total_samples = int(duration_s * 1e9)  # 1 ns per sample
    expected_interval = mean_time_between_pulses_s * 1e9

    pulse_times = []
    current_sample = 0

    while current_sample < total_samples:
        wait = np.random.exponential(expected_interval)
        current_sample += int(wait)
        if current_sample < total_samples:
            pulse_times.append(current_sample)

    return np.array(pulse_times, dtype=int)


def generate_scope_data(
        pulse_times,
        duration_s,
        noise_std=0.02,
        pulse_amplitude=2.0,
        pulse_duration_ns=10,
        rise_time_ns=2,
        filename="scope_data.csv"
    ):
    """
    Generate oscilloscope-like data at 1 ns sampling, given pulse times.
    Pulse times should be integer sample indices.
    """

    total_samples = int(duration_s * 1e9)
    t = np.arange(total_samples)

    # Background noise
    voltage = np.random.normal(0, noise_std, size=total_samples)

    pulse_width = pulse_duration_ns
    rise = rise_time_ns

    for pt in pulse_times:
        start = pt
        end = pt + pulse_width

        if start >= total_samples:
            continue
        if end >= total_samples:
            end = total_samples - 1

        # --- Pulse shape ---
        # Rise
        ramp_end = min(start + rise, total_samples)
        if ramp_end > start:
            voltage[start:ramp_end] += np.linspace(0, pulse_amplitude, ramp_end - start)

        # Flat top
        flat_start = ramp_end
        flat_end = min(start + pulse_width - rise, total_samples)
        if flat_end > flat_start:
            voltage[flat_start:flat_end] += pulse_amplitude

        # Fall
        fall_start = flat_end
        fall_end = min(fall_start + rise, total_samples)
        if fall_end > fall_start:
            voltage[fall_start:fall_end] += np.linspace(pulse_amplitude, 0, fall_end - fall_start)

    # Save CSV
    df = pd.DataFrame({"time_ns": t, "voltage": voltage})
    df.to_csv(filename, index=False)

    print(f"Saved {total_samples:,} samples to {filename}")


# -------------
# Example usage 
# -------------
duration = 0.02  # 20 milliseconds

# Step 1: generate pulse times
pulses = generate_pulse_times(duration_s=duration, mean_time_between_pulses_s=0.01)

# Step 2: generate the oscilloscope data file
generate_scope_data(
    pulse_times=pulses,
    duration_s=duration,
    noise_std=0.01,
    filename="scope_with_pulses.csv"
)

print("Done.")
