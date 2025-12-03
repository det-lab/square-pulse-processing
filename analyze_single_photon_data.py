import numpy as np
import pandas as pd
import RigolWFM.wfm as rigol

def count_pulses_in_interval(df, interval_length, threshold=0.5):
    """
    Count pulses inside equally sized time slices, ensuring that only full
    intervals are counted (i.e., intervals whose duration == interval_length).

    df: DataFrame with columns ["time_s", "voltage_V"]
    interval_length: size of each full interval in seconds
    threshold: rising-edge threshold for pulse detection (V)
    """
    times = df["time_s"].values
    volts = df["voltage_V"].values

    dt = times[1] - times[0]     # assumes uniform sampling
    n_samples_per_interval = int(round(interval_length / dt))

    # Compute number of *full* intervals only
    total_samples = len(times)
    n_full_intervals = total_samples // n_samples_per_interval

    if n_full_intervals == 0:
        raise ValueError("Data does not contain even one full interval.")

    interval_starts = []
    interval_ends = []
    pulse_counts = []

    # Loop over only the *full* intervals
    for i in range(n_full_intervals):
        start_idx = i * n_samples_per_interval
        end_idx   = (i + 1) * n_samples_per_interval

        # Extract slices for this interval
        t_slice = times[start_idx:end_idx]
        v_slice = volts[start_idx:end_idx]

        # Count pulses: rising-edge detection above threshold
        # (same logic as before)
        rising_edges = (v_slice[:-1] < threshold) & (v_slice[1:] >= threshold)
        pulse_count = rising_edges.sum()

        interval_starts.append(t_slice[0])
        interval_ends.append(t_slice[-1])
        pulse_counts.append(pulse_count)

    # Build the output DataFrame
    return pd.DataFrame({
        "interval_start_s": interval_starts,
        "interval_end_s": interval_ends,
        "pulse_count": pulse_counts
    })

def count_pulses_in_interval_multiple_files(filelist, interval_length, threshold=0.5):
    """
    Count pulses in equally sized time slices for multiple WFM files.

    Parameters
    ----------
    filelist : list of str
        List of .wfm filenames.
    interval_length : float
        Size of each interval (seconds).
    threshold : float
        Rising-edge threshold for pulse detection (volts).

    Returns
    -------
    DataFrame
        Columns:
            filename
            interval_start_s
            interval_end_s
            pulse_count
    """

    all_results = []   # store DataFrames to concatenate later

    for filename in filelist:
        # ---- Load waveform ----
        scope_data = rigol.Wfm.from_file(filename, '1000Z')
        ch = scope_data.channels[0]

        # ---- Check timestep ----
        dt = ch.times[1] - ch.times[0]
        if not np.isclose(dt, 1e-9):
            raise ValueError(f"{filename}: timestep {dt} != 1e-9")

        # ---- Build DataFrame for one file ----
        df = pd.DataFrame({
            "time_s": ch.times,
            "voltage_V": ch.volts
        })

        # ---- Count pulses in intervals ----
        interval_results = count_pulses_in_interval(
            df,
            interval_length=interval_length,
            threshold=threshold
        )

        # ---- Add filename column ----
        interval_results["filename"] = filename

        # ---- Save results ----
        all_results.append(interval_results)

    # ---- Combine all files into one DataFrame ----
    return pd.concat(all_results, ignore_index=True)