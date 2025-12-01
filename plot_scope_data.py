import pandas as pd
import matplotlib.pyplot as plt

def load_and_plot(filename="scope_data.csv"):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Plot the data
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_s"], df["voltage_V"], linewidth=0.7)
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")
    plt.title(filename)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


load_and_plot("scope_data.csv")

print("Done.")
