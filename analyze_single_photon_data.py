import numpy as np
import pandas as pd
import RigolWFM.wfm as rigol

def count_pulses_in_interval(df, interval_length, threshold=0.5):
    """
    Count pulses inside equally sized time slices.

    df: DataFrame with columns ['time_s', 'voltage_V']
    interval_length: size of each time slice in seconds
    threshold: rising-edge threshold (in volts) for pulse detection

    Returns a DataFrame with columns:
        interval_start_s
        interval_end_s
        pulse_count
    """

    times = df["time_s"].values
    volts = df["voltage_V"].values

    # -------- Detect pulse start times --------
    # A pulse occurs when voltage crosses upward through threshold
    above = volts > threshold
    rising_edges = np.where((~above[:-1]) & (above[1:]))[0]
    pulse_times = times[rising_edges]

    # -------- Bin edges for slicing --------
    t_min = times.min()
    t_max = times.max()

    bins = np.arange(t_min, t_max + interval_length, interval_length)

    # Digitize pulse times into bins
    bin_indices = np.digitize(pulse_times, bins) - 1  # shift to 0-based

    # Count pulses per interval
    pulse_counts = np.bincount(bin_indices, minlength=len(bins)-1)

    # -------- Build output DataFrame --------
    result = pd.DataFrame({
        "interval_start_s": bins[:-1],
        "interval_end_s": bins[1:],
        "pulse_count": pulse_counts
    })

    return result

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